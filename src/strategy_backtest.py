"""
Strategy backtest module.

Backtests the rolling total return signal strategy:
- OWN MSTY when all 3 timeframes (5D/10D/20D) show MSTY positive + WNTR negative
- OWN WNTR when all 3 timeframes show MSTY negative + WNTR positive
- CASH (or hold previous) when signals are mixed

Uses total return indices to correctly handle splits and dividends.
The TRI compounds price returns + dividend yield, so any period's return
can be computed as (TRI_end / TRI_start - 1) regardless of splits.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .counterparty_analysis import TotalReturnCalculator, CounterpartyAnalyzer


@dataclass
class Trade:
    """A single trade in the strategy."""
    trade_num: int
    instrument: str  # "MSTY" or "WNTR"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_tri: float  # Total return index at entry
    exit_tri: float   # Total return index at exit
    return_pct: float  # (exit_tri / entry_tri - 1) * 100
    holding_days: int
    entry_price: float
    exit_price: float
    signal_strength: int  # Number of timeframes agreeing (1-3)


@dataclass
class StrategyResult:
    """Complete strategy backtest result."""
    trades: List[Trade]
    daily_equity: pd.Series
    total_return_pct: float
    num_trades: int
    win_rate: float
    avg_holding_days: float
    msty_buy_hold_return: float
    wntr_buy_hold_return: float
    msty_tri: pd.Series
    wntr_tri: pd.Series


class StrategyBacktester:
    """
    Backtests the 3-timeframe divergence signal strategy.

    Signal rules:
    - STRONG OWN WNTR: All 3 timeframes diverging (MSTY neg + WNTR pos)
    - STRONG OWN MSTY: All 3 timeframes show MSTY pos + WNTR neg
    - MIXED: Anything else -> hold cash or previous position

    The strategy uses total return indices for P&L calculation,
    which correctly handles stock splits and dividend reinvestment.
    """

    def __init__(self, windows: List[int] = None, signal_threshold: int = 3):
        self.windows = windows or [5, 10, 20]
        self.signal_threshold = signal_threshold  # Min TFs agreeing to trade

    def classify_signal(self, row: pd.Series) -> str:
        """
        Classify the signal for a single day.

        Returns: "OWN_MSTY", "OWN_WNTR", or "NEUTRAL"
        """
        msty_pos_count = 0
        wntr_neg_count = 0
        msty_neg_count = 0
        wntr_pos_count = 0

        for w in self.windows:
            msty_col = f"msty_total_return_{w}d"
            wntr_col = f"wntr_total_return_{w}d"

            msty_ret = row.get(msty_col, np.nan)
            wntr_ret = row.get(wntr_col, np.nan)

            if pd.isna(msty_ret) or pd.isna(wntr_ret):
                return "NEUTRAL"

            if msty_ret > 0:
                msty_pos_count += 1
            else:
                msty_neg_count += 1

            if wntr_ret > 0:
                wntr_pos_count += 1
            else:
                wntr_neg_count += 1

        # OWN MSTY: MSTY positive on all TFs AND WNTR negative on all TFs
        if msty_pos_count >= self.signal_threshold and wntr_neg_count >= self.signal_threshold:
            return "OWN_MSTY"

        # OWN WNTR: MSTY negative on all TFs AND WNTR positive on all TFs
        if msty_neg_count >= self.signal_threshold and wntr_pos_count >= self.signal_threshold:
            return "OWN_WNTR"

        return "NEUTRAL"

    def run_backtest(
        self,
        msty_df: pd.DataFrame,
        wntr_df: pd.DataFrame,
        neutral_action: str = "cash",  # "cash" or "hold"
    ) -> StrategyResult:
        """
        Run the full strategy backtest.

        Args:
            msty_df: MSTY DataFrame with close, dividend, split columns
            wntr_df: WNTR DataFrame with close, dividend, split columns
            neutral_action: What to do on neutral signal:
                "cash" = go to cash (no position)
                "hold" = hold previous position

        Returns:
            StrategyResult with trades, equity curve, and statistics
        """
        # Build comparison data with rolling returns
        analyzer = CounterpartyAnalyzer(windows=self.windows)
        comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)

        # Calculate total return indices
        calc = TotalReturnCalculator()
        msty_tri = calc.calculate_total_return_index(msty_df)
        wntr_tri = calc.calculate_total_return_index(wntr_df)

        # Align TRIs to comparison dates
        common_dates = comparison.index

        # Classify signal for each day
        signals = pd.Series("NEUTRAL", index=common_dates)
        for date in common_dates:
            if date in comparison.index:
                signals[date] = self.classify_signal(comparison.loc[date])

        # Build daily equity curve
        # Start with $10,000
        initial_capital = 10000.0
        equity = pd.Series(np.nan, index=common_dates)
        equity.iloc[0] = initial_capital

        current_position = "CASH"  # CASH, MSTY, or WNTR
        trades = []
        trade_num = 0

        # Track current trade entry
        trade_entry_date = None
        trade_entry_tri = None
        trade_entry_price = None
        trade_instrument = None

        # For equity tracking, we need the TRI on the first valid signal day
        # and accumulate returns from position changes
        accumulated_equity = initial_capital

        for i, date in enumerate(common_dates):
            signal = signals[date]

            # Map signal to desired position
            if signal == "OWN_MSTY":
                desired_position = "MSTY"
            elif signal == "OWN_WNTR":
                desired_position = "WNTR"
            else:
                if neutral_action == "hold":
                    desired_position = current_position
                else:
                    desired_position = "CASH"

            # Check for position change
            if desired_position != current_position:
                # Close existing position
                if current_position in ("MSTY", "WNTR") and trade_entry_date is not None:
                    if current_position == "MSTY" and date in msty_tri.index:
                        exit_tri = msty_tri[date]
                        exit_price = msty_df.loc[date, "close"] if date in msty_df.index else np.nan
                    elif current_position == "WNTR" and date in wntr_tri.index:
                        exit_tri = wntr_tri[date]
                        exit_price = wntr_df.loc[date, "close"] if date in wntr_df.index else np.nan
                    else:
                        exit_tri = trade_entry_tri
                        exit_price = trade_entry_price

                    trade_return = (exit_tri / trade_entry_tri - 1) * 100
                    holding_days = (date - trade_entry_date).days

                    trade_num += 1
                    trades.append(Trade(
                        trade_num=trade_num,
                        instrument=current_position,
                        entry_date=trade_entry_date,
                        exit_date=date,
                        entry_tri=trade_entry_tri,
                        exit_tri=exit_tri,
                        return_pct=trade_return,
                        holding_days=holding_days,
                        entry_price=trade_entry_price,
                        exit_price=exit_price,
                        signal_strength=self.signal_threshold,
                    ))

                    # Update accumulated equity
                    accumulated_equity *= (1 + trade_return / 100)

                # Open new position
                if desired_position in ("MSTY", "WNTR"):
                    if desired_position == "MSTY" and date in msty_tri.index:
                        trade_entry_tri = msty_tri[date]
                        trade_entry_price = msty_df.loc[date, "close"] if date in msty_df.index else np.nan
                    elif desired_position == "WNTR" and date in wntr_tri.index:
                        trade_entry_tri = wntr_tri[date]
                        trade_entry_price = wntr_df.loc[date, "close"] if date in wntr_df.index else np.nan
                    else:
                        trade_entry_tri = None
                        trade_entry_price = np.nan

                    trade_entry_date = date
                    trade_instrument = desired_position
                else:
                    trade_entry_date = None
                    trade_entry_tri = None
                    trade_entry_price = np.nan

                current_position = desired_position

            # Calculate current equity
            if current_position == "MSTY" and trade_entry_tri and date in msty_tri.index:
                current_tri = msty_tri[date]
                period_return = current_tri / trade_entry_tri - 1
                equity[date] = accumulated_equity * (1 + period_return)
            elif current_position == "WNTR" and trade_entry_tri and date in wntr_tri.index:
                current_tri = wntr_tri[date]
                period_return = current_tri / trade_entry_tri - 1
                equity[date] = accumulated_equity * (1 + period_return)
            else:
                equity[date] = accumulated_equity

        # Close final position if still open
        if current_position in ("MSTY", "WNTR") and trade_entry_date is not None:
            final_date = common_dates[-1]
            if current_position == "MSTY" and final_date in msty_tri.index:
                exit_tri = msty_tri[final_date]
                exit_price = msty_df.loc[final_date, "close"] if final_date in msty_df.index else np.nan
            elif current_position == "WNTR" and final_date in wntr_tri.index:
                exit_tri = wntr_tri[final_date]
                exit_price = wntr_df.loc[final_date, "close"] if final_date in wntr_df.index else np.nan
            else:
                exit_tri = trade_entry_tri
                exit_price = trade_entry_price

            trade_return = (exit_tri / trade_entry_tri - 1) * 100
            holding_days = (final_date - trade_entry_date).days

            trade_num += 1
            trades.append(Trade(
                trade_num=trade_num,
                instrument=current_position,
                entry_date=trade_entry_date,
                exit_date=final_date,
                entry_tri=trade_entry_tri,
                exit_tri=exit_tri,
                return_pct=trade_return,
                holding_days=holding_days,
                entry_price=trade_entry_price,
                exit_price=exit_price,
                signal_strength=self.signal_threshold,
            ))

        # Forward-fill equity for any NaN days
        equity = equity.ffill()

        # Calculate buy-and-hold returns
        common_msty_tri = msty_tri.reindex(common_dates).dropna()
        common_wntr_tri = wntr_tri.reindex(common_dates).dropna()

        msty_bh = (common_msty_tri.iloc[-1] / common_msty_tri.iloc[0] - 1) * 100
        wntr_bh = (common_wntr_tri.iloc[-1] / common_wntr_tri.iloc[0] - 1) * 100

        # Strategy statistics
        total_return = (equity.iloc[-1] / initial_capital - 1) * 100
        winning_trades = [t for t in trades if t.return_pct > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_holding = np.mean([t.holding_days for t in trades]) if trades else 0

        return StrategyResult(
            trades=trades,
            daily_equity=equity,
            total_return_pct=total_return,
            num_trades=len(trades),
            win_rate=win_rate,
            avg_holding_days=avg_holding,
            msty_buy_hold_return=msty_bh,
            wntr_buy_hold_return=wntr_bh,
            msty_tri=common_msty_tri,
            wntr_tri=common_wntr_tri,
        )


class StrategyVisualizer:
    """Visualization for strategy backtest results."""

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_strategy_equity_curve(
        self,
        result: StrategyResult,
        comparison: pd.DataFrame,
    ) -> plt.Figure:
        """
        4-panel strategy performance chart:
        1. Strategy equity vs buy-and-hold alternatives
        2. Position map (what was held when)
        3. Per-trade P&L waterfall
        4. Drawdown comparison
        """
        fig, axes = plt.subplots(4, 1, figsize=(18, 22))

        initial = 10000.0

        # --- Panel 1: Equity curves ---
        ax1 = axes[0]

        # Strategy
        ax1.plot(result.daily_equity.index, result.daily_equity,
                 color="green", linewidth=2.5, label=f"Strategy ({result.total_return_pct:+.1f}%)")

        # MSTY buy-and-hold
        msty_equity = initial * (result.msty_tri / result.msty_tri.iloc[0])
        ax1.plot(msty_equity.index, msty_equity,
                 color="red", linewidth=1.5, alpha=0.7,
                 label=f"MSTY Buy & Hold ({result.msty_buy_hold_return:+.1f}%)")

        # WNTR buy-and-hold
        wntr_equity = initial * (result.wntr_tri / result.wntr_tri.iloc[0])
        ax1.plot(wntr_equity.index, wntr_equity,
                 color="blue", linewidth=1.5, alpha=0.7,
                 label=f"WNTR Buy & Hold ({result.wntr_buy_hold_return:+.1f}%)")

        # Cash line
        ax1.axhline(y=initial, color="gray", linewidth=1, linestyle="--",
                     alpha=0.5, label="Cash ($10,000)")

        ax1.set_title("Strategy Equity Curve vs Buy-and-Hold Alternatives",
                       fontweight="bold", fontsize=13)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Position map ---
        ax2 = axes[1]

        # Color-code the position held each day
        for trade in result.trades:
            color = "red" if trade.instrument == "MSTY" else "blue"
            alpha = 0.6 if trade.return_pct > 0 else 0.3
            ax2.axvspan(trade.entry_date, trade.exit_date,
                        alpha=alpha, color=color)

        ax2.set_title("Position Map — Red = MSTY, Blue = WNTR, White = Cash",
                       fontweight="bold", fontsize=11)
        ax2.set_ylabel("Position")
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, axis="x")

        # Add trade number labels
        for trade in result.trades:
            mid_date = trade.entry_date + (trade.exit_date - trade.entry_date) / 2
            ax2.text(mid_date, 0.5, f"#{trade.trade_num}\n{trade.instrument}\n{trade.return_pct:+.1f}%",
                     ha="center", va="center", fontsize=6, fontweight="bold",
                     transform=ax2.get_xaxis_transform())

        # --- Panel 3: Per-trade P&L waterfall ---
        ax3 = axes[2]

        if result.trades:
            trade_nums = [t.trade_num for t in result.trades]
            returns = [t.return_pct for t in result.trades]
            colors = ["green" if r > 0 else "red" for r in returns]
            instruments = [t.instrument for t in result.trades]

            bars = ax3.bar(trade_nums, returns, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

            # Label each bar
            for i, (tn, ret, inst) in enumerate(zip(trade_nums, returns, instruments)):
                ax3.text(tn, ret + (0.5 if ret >= 0 else -0.5),
                         f"{inst}\n{ret:+.1f}%",
                         ha="center", va="bottom" if ret >= 0 else "top",
                         fontsize=7, fontweight="bold")

        ax3.axhline(y=0, color="black", linewidth=1)
        ax3.set_title("Per-Trade Return (%)", fontweight="bold", fontsize=11)
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("Return (%)")
        ax3.grid(True, alpha=0.3, axis="y")

        # --- Panel 4: Drawdown comparison ---
        ax4 = axes[3]

        for label, equity_series, color in [
            ("Strategy", result.daily_equity, "green"),
            ("MSTY B&H", msty_equity, "red"),
            ("WNTR B&H", wntr_equity, "blue"),
        ]:
            running_max = equity_series.expanding().max()
            drawdown = (equity_series / running_max - 1) * 100
            ax4.plot(drawdown.index, drawdown, color=color, linewidth=1.5,
                     alpha=0.8, label=f"{label} (max: {drawdown.min():.1f}%)")
            ax4.fill_between(drawdown.index, drawdown, 0,
                             alpha=0.1, color=color)

        ax4.set_title("Drawdown Comparison", fontweight="bold", fontsize=11)
        ax4.set_ylabel("Drawdown (%)")
        ax4.set_xlabel("Date")
        ax4.legend(loc="best", fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "strategy_backtest.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig

    def plot_signal_timeline(
        self,
        comparison: pd.DataFrame,
        result: StrategyResult,
        windows: List[int] = None,
    ) -> plt.Figure:
        """
        3-panel signal timeline:
        1. MSTY & WNTR prices with trade entry/exit markers
        2. Divergence count over time with position overlay
        3. Strategy equity with trade annotations
        """
        windows = windows or [5, 10, 20]
        fig, axes = plt.subplots(3, 1, figsize=(18, 16))

        # --- Panel 1: Prices with trade markers ---
        ax1 = axes[0]

        if "msty_price" in comparison.columns:
            ax1.plot(comparison.index, comparison["msty_price"],
                     color="red", linewidth=1.2, alpha=0.7, label="MSTY Price")

        ax1_r = ax1.twinx()
        if "wntr_price" in comparison.columns:
            ax1_r.plot(comparison.index, comparison["wntr_price"],
                       color="blue", linewidth=1.2, alpha=0.7, label="WNTR Price")

        # Mark trade entries and exits
        for trade in result.trades:
            color = "red" if trade.instrument == "MSTY" else "blue"
            marker_ax = ax1 if trade.instrument == "MSTY" else ax1_r
            marker_ax.axvline(trade.entry_date, color=color, linewidth=0.8,
                              linestyle="--", alpha=0.5)

        ax1.set_title("Price Action with Trade Entry/Exit Points",
                       fontweight="bold", fontsize=12)
        ax1.set_ylabel("MSTY Price ($)", color="red")
        ax1_r.set_ylabel("WNTR Price ($)", color="blue")
        ax1.legend(loc="upper left", fontsize=9)
        ax1_r.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Divergence + position ---
        ax2 = axes[1]

        if "divergence_count" in comparison.columns:
            div_count = comparison["divergence_count"].dropna()
            colors_map = {0: "green", 1: "gold", 2: "orange", 3: "red"}
            ax2.bar(div_count.index, div_count,
                    color=[colors_map.get(int(v), "gray") for v in div_count],
                    width=1.5, alpha=0.7)

        # Overlay position
        for trade in result.trades:
            color = "red" if trade.instrument == "MSTY" else "blue"
            ax2.axvspan(trade.entry_date, trade.exit_date,
                        ymin=0.85, ymax=1.0, alpha=0.6, color=color)

        ax2.set_title("Divergence Count + Position Overlay (top bar)",
                       fontweight="bold", fontsize=11)
        ax2.set_ylabel("# Timeframes Diverging")
        ax2.set_yticks(range(len(windows) + 1))
        ax2.grid(True, alpha=0.3, axis="y")

        # --- Panel 3: Strategy equity ---
        ax3 = axes[2]

        ax3.plot(result.daily_equity.index, result.daily_equity,
                 color="green", linewidth=2, label="Strategy Equity")
        ax3.axhline(y=10000, color="gray", linewidth=1, linestyle="--", alpha=0.5)

        # Mark winners and losers
        for trade in result.trades:
            color_dot = "lime" if trade.return_pct > 0 else "red"
            if trade.exit_date in result.daily_equity.index:
                ax3.plot(trade.exit_date, result.daily_equity[trade.exit_date],
                         "o", color=color_dot, markersize=6, zorder=5)

        ax3.set_title("Strategy Equity Curve with Trade Outcomes",
                       fontweight="bold", fontsize=11)
        ax3.set_ylabel("Portfolio Value ($)")
        ax3.set_xlabel("Date")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "strategy_signal_timeline.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig
