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
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from .config import RISK, STRATEGY, MARKET_MAKER
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


class RiskManager:
    """
    Portfolio-level risk management.

    Usage:
        rm = RiskManager()
        report = rm.assess(execution_state)
        allowed, reason = rm.pre_trade_check(signal, execution_state)
        rm.adjust_size(signal, execution_state) → adjusted_size
    """

    def __init__(self):
        self.session_peak_pnl = 0.0
        self.kill_switch = False
        self._kill_switch_reason = ""

    def assess(self, state: ExecutionState) -> RiskReport:
        """
        Generate a risk report for the current portfolio state.
        """
        open_positions = [p for p in state.positions if p.status == "open"]
        total_exposure = sum(p.entry_cost for p in open_positions)
        max_exposure = RISK["max_total_exposure_usdc"]
        utilization = (total_exposure / max_exposure * 100) if max_exposure > 0 else 0

        # Track session peak
        if state.daily_pnl > self.session_peak_pnl:
            self.session_peak_pnl = state.daily_pnl

        drawdown = self.session_peak_pnl - state.daily_pnl

        warnings = []

        # High utilization warning
        if utilization > 80:
            warnings.append(f"High exposure utilization: {utilization:.0f}%")

        # Approaching daily loss limit
        remaining = RISK["daily_loss_limit_usdc"] + state.daily_pnl  # daily_pnl is negative when losing
        if remaining < RISK["daily_loss_limit_usdc"] * 0.25:
            warnings.append(f"Approaching daily loss limit: ${remaining:.2f} remaining")

        # Drawdown warning
        if drawdown > RISK["daily_loss_limit_usdc"] * 0.5:
            warnings.append(f"Significant drawdown from peak: ${drawdown:.2f}")

        # Kill switch conditions
        if state.daily_pnl <= -RISK["daily_loss_limit_usdc"]:
            self.kill_switch = True
            self._kill_switch_reason = "Daily loss limit exceeded"
            warnings.append("KILL SWITCH ACTIVATED: Daily loss limit exceeded")

        if drawdown > RISK["daily_loss_limit_usdc"] * 0.75:
            self.kill_switch = True
            self._kill_switch_reason = "Severe drawdown from session peak"
            warnings.append("KILL SWITCH ACTIVATED: Severe drawdown")

        return RiskReport(
            timestamp=time.time(),
            total_exposure=total_exposure,
            max_exposure=max_exposure,
            utilization_pct=utilization,
            open_positions=len(open_positions),
            max_positions=RISK["max_open_positions"],
            daily_pnl=state.daily_pnl,
            daily_loss_limit=RISK["daily_loss_limit_usdc"],
            daily_pnl_pct=(state.daily_pnl / max_exposure * 100) if max_exposure > 0 else 0,
            drawdown_from_peak=drawdown,
            kill_switch_active=self.kill_switch,
            warnings=warnings,
        )

    def pre_trade_check(self, signal: Signal,
                        state: ExecutionState) -> tuple[bool, str]:
        """
        Pre-trade risk validation beyond the basic execution checks.

        Returns (allowed, reason).
        """
        if self.kill_switch:
            return False, f"Kill switch active: {self._kill_switch_reason}"

        # Check correlation: don't stack too many positions on the same asset
        # Market maker gets its own higher limit — it's hedged (both sides)
        open_positions = [p for p in state.positions if p.status == "open"]
        same_asset = [p for p in open_positions if p.asset == signal.asset]
        is_mm = getattr(signal, "strategy", "") == "market_maker"
        max_per_asset = MARKET_MAKER["max_positions"] if is_mm else 2
        if len(same_asset) >= max_per_asset:
            return False, f"Too many positions on {signal.asset} ({len(same_asset)}/{max_per_asset})"

        # MM-specific exposure cap
        if is_mm:
            mm_exposure = sum(
                p.entry_cost for p in open_positions
                if getattr(p, "strategy", "") == "market_maker"
            )
            if mm_exposure >= MARKET_MAKER["max_exposure_usdc"]:
                return False, f"MM exposure cap reached (${mm_exposure:.0f}/${MARKET_MAKER['max_exposure_usdc']:.0f})"

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
            exposure_ratio = current_exposure / RISK["max_total_exposure_usdc"]

            if exposure_ratio > 0.5 and signal.edge < STRATEGY["min_edge"] * 1.5:
                return False, (f"Edge too low for current exposure "
                               f"({signal.edge:.3f} < {STRATEGY['min_edge'] * 1.5:.3f})")

        return True, "OK"

    def adjust_size(self, signal: Signal,
                    state: ExecutionState) -> float:
        """
        Adjust position size based on portfolio context.

        Returns the adjusted size (number of shares).
        """
        base_size = signal.suggested_size

        # Scale down as we approach max exposure
        open_positions = [p for p in state.positions if p.status == "open"]
        current_exposure = sum(p.entry_cost for p in open_positions)
        remaining_capacity = RISK["max_total_exposure_usdc"] - current_exposure

        max_cost = min(
            RISK["max_position_usdc"],
            remaining_capacity,
        )

        # Cost of the base size
        base_cost = base_size * signal.suggested_price
        if base_cost > max_cost:
            base_size = max(1, int(max_cost / signal.suggested_price))

        # Scale by confidence
        if signal.confidence < 0.5:
            base_size = max(1, int(base_size * 0.5))

        # Scale down after losses
        if state.daily_pnl < 0:
            loss_ratio = abs(state.daily_pnl) / RISK["daily_loss_limit_usdc"]
            scale = max(0.25, 1 - loss_ratio)
            base_size = max(1, int(base_size * scale))

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
