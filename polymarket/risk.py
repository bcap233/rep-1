"""
Risk Management Module.

Provides additional risk analysis and portfolio-level controls
beyond the per-trade limits in the execution engine.

Responsibilities:
  - Portfolio-level exposure tracking
  - Correlation-aware position sizing
  - Volatility-adjusted sizing
  - Drawdown monitoring
  - Kill switch for anomalous conditions
  - Kelly criterion bankroll management
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import RISK, STRATEGY, MARKET_MAKER, HIGH_PROB_GRINDER, KELLY
from .execution import Position, ExecutionState
from .signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskReport:
    """Portfolio risk snapshot."""
    timestamp: float
    total_exposure: float
    max_exposure: float
    utilization_pct: float      # exposure / max as %
    open_positions: int
    max_positions: int
    daily_pnl: float
    daily_loss_limit: float
    daily_pnl_pct: float        # pnl / max_exposure as %
    drawdown_from_peak: float   # Current drawdown from session peak
    kill_switch_active: bool
    warnings: list[str]


class BankrollManager:
    """
    Kelly-criterion bankroll management.

    Tracks cumulative PnL across sessions and computes a dynamic scale
    factor for position sizing. Uses half-Kelly by default for
    sustainable compounding with controlled variance.

    The scale factor multiplies base risk limits:
      - max_total_exposure_usdc
      - max_position_usdc
      - daily_loss_limit_usdc
      - MM max_exposure_usdc
    """

    def __init__(self):
        self.initial_bankroll = KELLY["initial_bankroll_usdc"]
        self.bankroll = self.initial_bankroll
        self.peak_bankroll = self.initial_bankroll
        self.cumulative_pnl = 0.0
        self.trade_history: list[float] = []  # list of realized PnLs
        self._load()
        self._reconcile_from_trade_log()

    # ---- Persistence ----

    def _load(self):
        path = Path(KELLY["bankroll_file"])
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.bankroll = data.get("bankroll", self.initial_bankroll)
                self.peak_bankroll = data.get("peak_bankroll", self.bankroll)
                self.cumulative_pnl = data.get("cumulative_pnl", 0.0)
                self.trade_history = data.get("trade_history", [])
                # Keep only the rolling window
                window = KELLY["rolling_window"]
                self.trade_history = self.trade_history[-window:]
                logger.info(f"[KELLY] Loaded bankroll: ${self.bankroll:.2f} "
                            f"(peak ${self.peak_bankroll:.2f}, "
                            f"{len(self.trade_history)} trades in window)")
            except Exception as e:
                logger.warning(f"[KELLY] Could not load bankroll: {e}")

    def _reconcile_from_trade_log(self):
        """Reconcile cumulative_pnl with the trade log (source of truth).

        The trade log records every close independently. If there's a gap
        (e.g., bankroll tracking started late, or a crash lost updates),
        this corrects cumulative_pnl and bankroll on startup.

        Memory-optimised: pre-filters lines with string checks before
        parsing JSON, and only extracts the two fields needed (action, pnl).
        """
        from .config import DATA
        trade_log_path = Path(DATA["trade_log"])
        if not trade_log_path.exists():
            return

        try:
            log_total = 0.0
            with open(trade_log_path) as f:
                for line in f:
                    # Fast pre-filter: skip lines that aren't CLOSE actions
                    # without paying the cost of full JSON parsing.
                    if '"CLOSE"' not in line:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("action") == "CLOSE":
                        log_total += entry.get("pnl", 0)
                    del entry  # free immediately
        except Exception as e:
            logger.warning(f"[KELLY] Could not reconcile from trade log: {e}")
            return

        gap = log_total - self.cumulative_pnl
        if abs(gap) > 1.0:
            logger.info(f"[KELLY] Reconciling bankroll: trade_log=${log_total:+.2f} "
                        f"vs cumulative=${self.cumulative_pnl:+.2f} (gap=${gap:+.2f})")
            self.cumulative_pnl = log_total
            self.bankroll = self.initial_bankroll + log_total
            if self.bankroll > self.peak_bankroll:
                self.peak_bankroll = self.bankroll
            self.save()

    def save(self):
        path = Path(KELLY["bankroll_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bankroll": round(self.bankroll, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "trade_history": [round(p, 2) for p in self.trade_history[-KELLY["rolling_window"]:]],
            "last_update": time.time(),
        }
        # Atomic write: temp file + rename prevents corruption on crash
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- Core Kelly math ----

    def record_trade(self, realized_pnl: float):
        """Record a closed trade's PnL and update bankroll."""
        self.trade_history.append(realized_pnl)
        # Trim to rolling window
        window = KELLY["rolling_window"]
        if len(self.trade_history) > window:
            self.trade_history = self.trade_history[-window:]

        self.cumulative_pnl += realized_pnl
        self.bankroll += realized_pnl

        # Update peak
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        self.save()

    def kelly_fraction(self) -> float:
        """
        Compute Kelly fraction from rolling trade history.

        Kelly f* = (p * avg_win - q * avg_loss) / avg_win
        where p = win rate, q = 1-p

        Returns 0.0 if not enough data or edge is negative.
        """
        if len(self.trade_history) < KELLY["min_trades_for_kelly"]:
            return 0.0

        wins = [p for p in self.trade_history if p > 0]
        losses = [abs(p) for p in self.trade_history if p < 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(self.trade_history)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_win == 0:
            return 0.0

        # Kelly formula
        f_star = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Clamp to [0, 1] — negative means don't bet at all
        return max(0.0, min(1.0, f_star))

    def scale_factor(self) -> float:
        """
        Compute the bankroll scale factor for position sizing.

        Base scale = bankroll / initial_bankroll (simple ratio).
        Modulated by Kelly: if Kelly says bet less, we scale down.
        Clamped between min_scale and max_scale.
        Subject to drawdown throttle.
        """
        # Base: how much has the bankroll grown?
        base = self.bankroll / self.initial_bankroll

        # Kelly modulation: blend base scale toward 1.0 based on Kelly
        kelly_f = self.kelly_fraction()
        fractional_kelly = kelly_f * KELLY["fraction"]

        if len(self.trade_history) >= KELLY["min_trades_for_kelly"]:
            # Scale = base adjusted by Kelly confidence
            # If Kelly says 0 (no edge), scale stays at 1.0
            # If Kelly says high edge, allow full bankroll scaling
            if fractional_kelly > 0:
                # Use max(base, 1.0 + fractional_kelly) so scaling works
                # even when bankroll hasn't grown yet (base ~= 1.0)
                kelly_scale = 1.0 + fractional_kelly * 2  # Kelly-implied scale
                scale = max(kelly_scale, 1.0 + (base - 1.0) * min(1.0, fractional_kelly * 4))
            else:
                # Negative or zero edge: shrink back toward 1.0
                scale = 1.0
        else:
            # Not enough data yet — use conservative bankroll scaling
            scale = 1.0 + (base - 1.0) * 0.3  # only 30% of growth

        # Drawdown throttle
        if self.peak_bankroll > 0:
            drawdown_pct = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
            if drawdown_pct > KELLY["drawdown_throttle_pct"]:
                # In drawdown — scale back toward 1.0
                throttle = max(0.3, 1.0 - drawdown_pct * 3)
                scale *= throttle
                logger.info(f"[KELLY] Drawdown throttle active: "
                            f"{drawdown_pct:.1%} DD → {throttle:.2f}x throttle")

        # Clamp
        scale = max(KELLY["min_scale"], min(KELLY["max_scale"], scale))
        return scale

    def get_effective_limits(self) -> dict:
        """
        Return scaled risk limits based on current bankroll.

        These override the static config values.
        Hard per-trade cap is never scaled (absolute ceiling).
        """
        s = self.scale_factor()
        hard_cap = RISK.get("hard_max_per_trade_usdc", 150.0)
        return {
            "max_total_exposure_usdc": RISK["max_total_exposure_usdc"] * s,
            "max_position_usdc": min(RISK["max_position_usdc"] * s, hard_cap),
            "daily_loss_limit_usdc": RISK["daily_loss_limit_usdc"] * s,
            "mm_max_exposure_usdc": MARKET_MAKER["max_exposure_usdc"] * s,
        }

    def summary(self) -> dict:
        """Bankroll status for display."""
        return {
            "bankroll": round(self.bankroll, 2),
            "initial": round(self.initial_bankroll, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "peak": round(self.peak_bankroll, 2),
            "drawdown": round(self.peak_bankroll - self.bankroll, 2),
            "kelly_fraction": round(self.kelly_fraction(), 4),
            "half_kelly": round(self.kelly_fraction() * KELLY["fraction"], 4),
            "scale_factor": round(self.scale_factor(), 3),
            "trades_in_window": len(self.trade_history),
            "win_rate": round(
                len([p for p in self.trade_history if p > 0]) / len(self.trade_history), 3
            ) if self.trade_history else 0.0,
        }


class RiskManager:
    """
    Portfolio-level risk management.

    Usage:
        rm = RiskManager()
        report = rm.assess(execution_state)
        allowed, reason = rm.pre_trade_check(signal, execution_state)
        rm.adjust_size(signal, execution_state) → adjusted_size
    """

    def __init__(self, bankroll_mgr: Optional[BankrollManager] = None):
        self.session_peak_pnl = 0.0
        self.kill_switch = False
        self._kill_switch_reason = ""
        self.bankroll = bankroll_mgr or BankrollManager()

    def assess(self, state: ExecutionState) -> RiskReport:
        """
        Generate a risk report for the current portfolio state.
        Uses Kelly-scaled limits for all thresholds.
        """
        limits = self.bankroll.get_effective_limits()
        open_positions = [p for p in state.positions if p.status == "open"]
        total_exposure = sum(p.entry_cost for p in open_positions)
        max_exposure = limits["max_total_exposure_usdc"]
        utilization = (total_exposure / max_exposure * 100) if max_exposure > 0 else 0

        # Track session peak
        if state.daily_pnl > self.session_peak_pnl:
            self.session_peak_pnl = state.daily_pnl

        drawdown = self.session_peak_pnl - state.daily_pnl
        daily_loss_limit = limits["daily_loss_limit_usdc"]

        warnings = []

        # High utilization warning
        if utilization > 80:
            warnings.append(f"High exposure utilization: {utilization:.0f}%")

        # Approaching daily loss limit
        remaining = daily_loss_limit + state.daily_pnl  # daily_pnl is negative when losing
        if remaining < daily_loss_limit * 0.25:
            warnings.append(f"Approaching daily loss limit: ${remaining:.2f} remaining")

        # Drawdown warning — relative to bankroll, not daily loss limit
        bankroll_val = self.bankroll.bankroll if self.bankroll else RISK["max_total_exposure_usdc"]
        drawdown_pct = drawdown / bankroll_val if bankroll_val > 0 else 0
        if drawdown_pct > 0.05:
            warnings.append(f"Drawdown from peak: ${drawdown:.2f} ({drawdown_pct:.1%} of bankroll)")

        # Kill switch: only on absolute daily loss, not drawdowns from profit peaks.
        # A session that's up $1,000 and dips $150 should NOT be killed.
        if state.daily_pnl <= -daily_loss_limit:
            self.kill_switch = True
            self._kill_switch_reason = "Daily loss limit exceeded"
            warnings.append("KILL SWITCH ACTIVATED: Daily loss limit exceeded")

        # Kill switch on severe drawdown — only if daily PnL is negative
        # (i.e., we're losing, not just giving back some of our gains)
        if state.daily_pnl < 0 and drawdown > daily_loss_limit * 0.75:
            self.kill_switch = True
            self._kill_switch_reason = "Severe drawdown while in loss"
            warnings.append("KILL SWITCH ACTIVATED: Severe drawdown while in loss")

        return RiskReport(
            timestamp=time.time(),
            total_exposure=total_exposure,
            max_exposure=max_exposure,
            utilization_pct=utilization,
            open_positions=len(open_positions),
            max_positions=RISK["max_open_positions"],
            daily_pnl=state.daily_pnl,
            daily_loss_limit=daily_loss_limit,
            daily_pnl_pct=(state.daily_pnl / max_exposure * 100) if max_exposure > 0 else 0,
            drawdown_from_peak=drawdown,
            kill_switch_active=self.kill_switch,
            warnings=warnings,
        )

    def pre_trade_check(self, signal: Signal,
                        state: ExecutionState) -> tuple[bool, str]:
        """
        Pre-trade risk validation beyond the basic execution checks.
        Uses Kelly-scaled limits.

        Returns (allowed, reason).
        """
        if self.kill_switch:
            return False, f"Kill switch active: {self._kill_switch_reason}"

        limits = self.bankroll.get_effective_limits()

        open_positions = [p for p in state.positions if p.status == "open"]

        # Lock signals (buying the opposite side of a held position)
        # ALWAYS allowed — they convert directional risk into guaranteed
        # profit. A lock reduces risk, so bypass asset/exposure limits.
        is_lock = self._is_lock_signal(signal, open_positions)
        if is_lock:
            return True, "OK (lock — reduces risk)"

        # Check correlation: don't stack too many positions on the same asset.
        # MM and grinder get their own limits — they're designed for many
        # small positions. Other strategies default to 2.
        same_asset = [p for p in open_positions if p.asset == signal.asset]
        strategy = getattr(signal, "strategy", "")
        is_mm = strategy == "market_maker"
        if is_mm:
            max_per_asset = MARKET_MAKER["max_positions"]
        elif strategy == "high_prob_grinder":
            max_per_asset = HIGH_PROB_GRINDER.get("max_positions", 10)
        else:
            max_per_asset = 2
        if len(same_asset) >= max_per_asset:
            return False, f"Too many positions on {signal.asset} ({len(same_asset)}/{max_per_asset})"

        # MM-specific exposure cap (Kelly-scaled)
        if is_mm:
            mm_exposure = sum(
                p.entry_cost for p in open_positions
                if getattr(p, "strategy", "") == "market_maker"
            )
            mm_cap = limits["mm_max_exposure_usdc"]
            if mm_exposure >= mm_cap:
                return False, f"MM exposure cap reached (${mm_exposure:.0f}/${mm_cap:.0f})"

        # Check if we're in a losing streak (skip for MM — spread trades
        # have different loss characteristics than directional bets)
        if not is_mm:
            recent_closed = state.closed_positions[-5:] if state.closed_positions else []
            if len(recent_closed) >= 3:
                recent_losses = sum(1 for p in recent_closed if p.realized_pnl < 0)
                if recent_losses >= 3:
                    return False, "Losing streak (3+ consecutive losses)"

        # Edge quality check: require higher edge when we're already exposed
        # MM uses its own exposure cap above, so skip this check for MM
        if not is_mm:
            current_exposure = sum(p.entry_cost for p in open_positions)
            max_total = limits["max_total_exposure_usdc"]
            exposure_ratio = current_exposure / max_total if max_total > 0 else 0

            if exposure_ratio > 0.5 and signal.edge < STRATEGY["min_edge"] * 1.5:
                return False, (f"Edge too low for current exposure "
                               f"({signal.edge:.3f} < {STRATEGY['min_edge'] * 1.5:.3f})")

        return True, "OK"

    @staticmethod
    def _is_lock_signal(signal, open_positions) -> bool:
        """Check if a signal is locking an existing position.

        A lock buys the OPPOSITE side of a market we already hold.
        Example: we hold Up on condition ABC, now buying Down on ABC.
        This guarantees $1.00 payout = risk-free.
        """
        cond = signal.market.condition_id
        sig_side = signal.token_side

        # Bilateral arb legs (immediate lock) are always locks
        if getattr(signal, "is_bilateral", False):
            return True

        # Check if we hold the opposite side on the same condition
        for pos in open_positions:
            if pos.condition_id != cond or pos.status != "open":
                continue
            # Opposite sides: Up↔Down, YES↔NO
            if (pos.token_side == "Up" and sig_side == "Down" or
                    pos.token_side == "Down" and sig_side == "Up" or
                    pos.token_side == "YES" and sig_side == "NO" or
                    pos.token_side == "NO" and sig_side == "YES"):
                return True

        return False

    def adjust_size(self, signal: Signal,
                    state: ExecutionState) -> float:
        """
        Adjust position size based on portfolio context.

        MM book: sized by remaining capacity only. No confidence/loss
        scaling — spread trading has different loss characteristics
        than directional bets.

        Directional book (grinder, spot_div, etc.): scaled by
        confidence and daily loss ratio. Stops make sense here
        because you're taking a view.

        Returns the adjusted size (number of shares).
        """
        limits = self.bankroll.get_effective_limits()
        base_size = signal.suggested_size
        strategy = getattr(signal, "strategy", "")

        # Scale down as we approach max exposure (Kelly-scaled)
        open_positions = [p for p in state.positions if p.status == "open"]
        current_exposure = sum(p.entry_cost for p in open_positions)
        remaining_capacity = limits["max_total_exposure_usdc"] - current_exposure

        max_cost = min(
            limits["max_position_usdc"],
            remaining_capacity,
        )

        # Cost of the base size
        base_cost = base_size * signal.suggested_price
        if base_cost > max_cost:
            base_size = max(1, int(max_cost / signal.suggested_price))

        # --- Directional book: confidence + loss scaling ---
        if strategy != "market_maker":
            # Scale by confidence
            if signal.confidence < 0.5:
                base_size = max(1, int(base_size * 0.5))

            # Scale down after losses (using Kelly-scaled daily limit)
            if state.daily_pnl < 0:
                loss_ratio = abs(state.daily_pnl) / limits["daily_loss_limit_usdc"]
                scale = max(0.25, 1 - loss_ratio)
                base_size = max(1, int(base_size * scale))

        # MM book: no confidence/loss scaling — spread trades don't benefit
        # from shrinking after losses (loss is from inventory, not direction)

        return base_size

    def reset_kill_switch(self):
        """Manually reset the kill switch."""
        self.kill_switch = False
        self._kill_switch_reason = ""
        logger.info("Kill switch reset")

    def reset_session(self):
        """Reset session tracking (e.g., start of new trading day)."""
        self.session_peak_pnl = 0.0
        self.kill_switch = False
        self._kill_switch_reason = ""
        logger.info("Risk session reset")
