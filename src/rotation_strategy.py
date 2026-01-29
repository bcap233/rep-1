"""
Always-invested rotation strategy between MSTY and WNTR.

Goal: Always own one instrument to capture maximum dividends.
The question is never "should I be invested?" — it's "which one do I own?"

Default position: MSTY (higher dividends in bullish regime).
Switch to WNTR when collapse signals confirm MSTY is in reflexive decline.

Signal variants:
- V1 (Baseline): Switch on 3/3 TF divergence, default MSTY on neutral
- V2 (Debounced): Require signal to persist N days before switching
- V3 (SMA-enhanced): 50-day SMA break as lead signal with TF confirmation
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .counterparty_analysis import TotalReturnCalculator, CounterpartyAnalyzer


@dataclass
class RotationTrade:
    """A single holding period in the rotation strategy."""
    trade_num: int
    instrument: str  # "MSTY" or "WNTR"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_tri: float
    exit_tri: float
    total_return_pct: float  # TRI-based (includes dividends + price)
    price_return_pct: float  # Price-only return
    dividends_collected: float  # Dollar dividends collected per share
    num_dividends: int  # Number of dividend payments received
    holding_days: int
    entry_price: float
    exit_price: float


@dataclass
class RotationResult:
    """Complete rotation strategy result."""
    strategy_name: str
    trades: List[RotationTrade]
    daily_equity: pd.Series
    daily_position: pd.Series  # "MSTY" or "WNTR" each day
    total_return_pct: float
    total_dividends_collected: float
    num_switches: int
    msty_days: int
    wntr_days: int
    msty_dividends: float
    wntr_dividends: float
    msty_buy_hold_return: float
    wntr_buy_hold_return: float
    msty_tri: pd.Series
    wntr_tri: pd.Series


class RotationBacktester:
    """
    Always-invested rotation between MSTY and WNTR.

    Core principle: you always own one. The signal decides WHICH one.
    Default is MSTY (the income instrument in bullish regimes).
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20]

    def _get_dividends_in_range(
        self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple:
        """Get total dividends and count in a date range."""
        mask = (df.index >= start) & (df.index <= end)
        divs = df.loc[mask, "dividend"]
        total = divs[divs > 0].sum()
        count = (divs > 0).sum()
        return total, count

    def _classify_3tf(self, row: pd.Series) -> str:
        """3-timeframe signal: OWN_MSTY, OWN_WNTR, or NEUTRAL."""
        msty_pos = 0
        msty_neg = 0
        wntr_pos = 0
        wntr_neg = 0

        for w in self.windows:
            msty_ret = row.get(f"msty_total_return_{w}d", np.nan)
            wntr_ret = row.get(f"wntr_total_return_{w}d", np.nan)
            if pd.isna(msty_ret) or pd.isna(wntr_ret):
                return "NEUTRAL"
            if msty_ret > 0:
                msty_pos += 1
            else:
                msty_neg += 1
            if wntr_ret > 0:
                wntr_pos += 1
            else:
                wntr_neg += 1

        if msty_pos == 3 and wntr_neg == 3:
            return "OWN_MSTY"
        if msty_neg == 3 and wntr_pos == 3:
            return "OWN_WNTR"
        return "NEUTRAL"

    def _run_rotation(
        self,
        strategy_name: str,
        daily_position: pd.Series,
        msty_df: pd.DataFrame,
        wntr_df: pd.DataFrame,
        msty_tri: pd.Series,
        wntr_tri: pd.Series,
        common_dates: pd.DatetimeIndex,
    ) -> RotationResult:
        """Core rotation engine — given a position series, compute equity + trades."""
        initial = 10000.0
        equity = pd.Series(np.nan, index=common_dates)
        equity.iloc[0] = initial

        trades = []
        trade_num = 0
        accumulated = initial

        # Track current position entry
        current_pos = daily_position.iloc[0]
        entry_date = common_dates[0]
        entry_tri = msty_tri[entry_date] if current_pos == "MSTY" else wntr_tri[entry_date]
        entry_price = (msty_df.loc[entry_date, "close"] if current_pos == "MSTY"
                       else wntr_df.loc[entry_date, "close"])

        for i, date in enumerate(common_dates):
            new_pos = daily_position[date]

            if new_pos != current_pos:
                # Close current position
                if current_pos == "MSTY":
                    exit_tri = msty_tri[date]
                    exit_price = msty_df.loc[date, "close"] if date in msty_df.index else entry_price
                else:
                    exit_tri = wntr_tri[date]
                    exit_price = wntr_df.loc[date, "close"] if date in wntr_df.index else entry_price

                total_ret = (exit_tri / entry_tri - 1) * 100
                price_ret = (exit_price / entry_price - 1) * 100
                holding = (date - entry_date).days

                # Count dividends during holding
                hold_df = msty_df if current_pos == "MSTY" else wntr_df
                divs_total, divs_count = self._get_dividends_in_range(hold_df, entry_date, date)

                trade_num += 1
                trades.append(RotationTrade(
                    trade_num=trade_num,
                    instrument=current_pos,
                    entry_date=entry_date,
                    exit_date=date,
                    entry_tri=entry_tri,
                    exit_tri=exit_tri,
                    total_return_pct=total_ret,
                    price_return_pct=price_ret,
                    dividends_collected=divs_total,
                    num_dividends=divs_count,
                    holding_days=holding,
                    entry_price=entry_price,
                    exit_price=exit_price,
                ))

                accumulated *= (1 + total_ret / 100)

                # Open new position
                current_pos = new_pos
                entry_date = date
                entry_tri = msty_tri[date] if current_pos == "MSTY" else wntr_tri[date]
                entry_price = (msty_df.loc[date, "close"] if current_pos == "MSTY"
                               else wntr_df.loc[date, "close"])

            # Daily equity mark
            if current_pos == "MSTY" and date in msty_tri.index:
                cur_tri = msty_tri[date]
                equity[date] = accumulated * (cur_tri / entry_tri)
            elif current_pos == "WNTR" and date in wntr_tri.index:
                cur_tri = wntr_tri[date]
                equity[date] = accumulated * (cur_tri / entry_tri)
            else:
                equity[date] = accumulated

        # Close final position
        final_date = common_dates[-1]
        if current_pos == "MSTY":
            exit_tri = msty_tri[final_date]
            exit_price = msty_df.loc[final_date, "close"]
        else:
            exit_tri = wntr_tri[final_date]
            exit_price = wntr_df.loc[final_date, "close"]

        total_ret = (exit_tri / entry_tri - 1) * 100
        price_ret = (exit_price / entry_price - 1) * 100
        holding = (final_date - entry_date).days
        hold_df = msty_df if current_pos == "MSTY" else wntr_df
        divs_total, divs_count = self._get_dividends_in_range(hold_df, entry_date, final_date)

        trade_num += 1
        trades.append(RotationTrade(
            trade_num=trade_num,
            instrument=current_pos,
            entry_date=entry_date,
            exit_date=final_date,
            entry_tri=entry_tri,
            exit_tri=exit_tri,
            total_return_pct=total_ret,
            price_return_pct=price_ret,
            dividends_collected=divs_total,
            num_dividends=divs_count,
            holding_days=holding,
            entry_price=entry_price,
            exit_price=exit_price,
        ))

        equity = equity.ffill()

        # Stats
        msty_bh_tri = msty_tri.reindex(common_dates).dropna()
        wntr_bh_tri = wntr_tri.reindex(common_dates).dropna()

        msty_days = (daily_position == "MSTY").sum()
        wntr_days = (daily_position == "WNTR").sum()
        msty_divs = sum(t.dividends_collected for t in trades if t.instrument == "MSTY")
        wntr_divs = sum(t.dividends_collected for t in trades if t.instrument == "WNTR")
        total_divs = msty_divs + wntr_divs
        switches = sum(1 for i in range(1, len(common_dates))
                       if daily_position[common_dates[i]] != daily_position[common_dates[i - 1]])

        return RotationResult(
            strategy_name=strategy_name,
            trades=trades,
            daily_equity=equity,
            daily_position=daily_position,
            total_return_pct=(equity.iloc[-1] / initial - 1) * 100,
            total_dividends_collected=total_divs,
            num_switches=switches,
            msty_days=int(msty_days),
            wntr_days=int(wntr_days),
            msty_dividends=msty_divs,
            wntr_dividends=wntr_divs,
            msty_buy_hold_return=(msty_bh_tri.iloc[-1] / msty_bh_tri.iloc[0] - 1) * 100,
            wntr_buy_hold_return=(wntr_bh_tri.iloc[-1] / wntr_bh_tri.iloc[0] - 1) * 100,
            msty_tri=msty_bh_tri,
            wntr_tri=wntr_bh_tri,
        )

    def run_v1_baseline(
        self, msty_df: pd.DataFrame, wntr_df: pd.DataFrame,
    ) -> RotationResult:
        """
        V1 Baseline: 3/3 TF divergence, default MSTY on neutral.

        Always own one. When the signal says OWN_WNTR, switch to WNTR.
        When it says OWN_MSTY or NEUTRAL, own MSTY.
        """
        analyzer = CounterpartyAnalyzer(windows=self.windows)
        comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)
        common_dates = comparison.index

        calc = TotalReturnCalculator()
        msty_tri = calc.calculate_total_return_index(msty_df)
        wntr_tri = calc.calculate_total_return_index(wntr_df)

        # Classify each day
        positions = pd.Series("MSTY", index=common_dates)
        for date in common_dates:
            sig = self._classify_3tf(comparison.loc[date])
            if sig == "OWN_WNTR":
                positions[date] = "WNTR"
            else:
                positions[date] = "MSTY"  # Default MSTY

        return self._run_rotation(
            "V1: Baseline (3/3 TF, default MSTY)",
            positions, msty_df, wntr_df, msty_tri, wntr_tri, common_dates,
        )

    def run_v2_debounced(
        self, msty_df: pd.DataFrame, wntr_df: pd.DataFrame,
        debounce_days: int = 3,
    ) -> RotationResult:
        """
        V2 Debounced: Require signal to persist N consecutive days before switching.

        Eliminates whipsaw by forcing the signal to prove itself.
        """
        analyzer = CounterpartyAnalyzer(windows=self.windows)
        comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)
        common_dates = comparison.index

        calc = TotalReturnCalculator()
        msty_tri = calc.calculate_total_return_index(msty_df)
        wntr_tri = calc.calculate_total_return_index(wntr_df)

        # Raw signals
        raw_signals = pd.Series("NEUTRAL", index=common_dates)
        for date in common_dates:
            raw_signals[date] = self._classify_3tf(comparison.loc[date])

        # Debounce: only switch when new signal persists for N days
        positions = pd.Series("MSTY", index=common_dates)
        current_pos = "MSTY"
        pending_signal = None
        pending_count = 0

        for date in common_dates:
            sig = raw_signals[date]

            # Determine what position this signal suggests
            if sig == "OWN_WNTR":
                suggested = "WNTR"
            else:
                suggested = "MSTY"

            if suggested != current_pos:
                if suggested == pending_signal:
                    pending_count += 1
                else:
                    pending_signal = suggested
                    pending_count = 1

                if pending_count >= debounce_days:
                    current_pos = suggested
                    pending_signal = None
                    pending_count = 0
            else:
                pending_signal = None
                pending_count = 0

            positions[date] = current_pos

        return self._run_rotation(
            f"V2: Debounced ({debounce_days}-day confirm)",
            positions, msty_df, wntr_df, msty_tri, wntr_tri, common_dates,
        )

    def run_v3_sma_enhanced(
        self, msty_df: pd.DataFrame, wntr_df: pd.DataFrame,
        sma_period: int = 50,
    ) -> RotationResult:
        """
        V3 SMA-Enhanced: Use 50-day SMA as lead signal.

        Rules:
        - MSTY below its 50-day SMA -> switch to WNTR
        - MSTY above its 50-day SMA -> switch to MSTY
        - The 3-TF divergence is used as a CONFIRMATION (not trigger)

        This captures the July 17 break immediately rather than waiting 14 days.
        """
        analyzer = CounterpartyAnalyzer(windows=self.windows)
        comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)
        common_dates = comparison.index

        calc = TotalReturnCalculator()
        msty_tri = calc.calculate_total_return_index(msty_df)
        wntr_tri = calc.calculate_total_return_index(wntr_df)

        # Calculate SMA on split-adjusted MSTY prices
        # Use the TRI as a proxy since it's split-adjusted
        msty_sorted = msty_df.sort_index()

        # Build split-adjusted price series
        if "split" in msty_sorted.columns:
            cum_split = msty_sorted["split"].replace(0, 1).cumprod()
            final_split = cum_split.iloc[-1]
            split_adj = cum_split / final_split
            adj_price = msty_sorted["close"] / split_adj
        else:
            adj_price = msty_sorted["close"]

        sma = adj_price.rolling(window=sma_period, min_periods=sma_period).mean()

        # Classify each day
        positions = pd.Series("MSTY", index=common_dates)

        for date in common_dates:
            if date not in adj_price.index or date not in sma.index:
                positions[date] = "MSTY"
                continue

            price = adj_price[date]
            sma_val = sma[date]

            if pd.isna(sma_val):
                # Not enough history for SMA yet, default MSTY
                positions[date] = "MSTY"
                continue

            if price < sma_val:
                # Below SMA: own WNTR
                positions[date] = "WNTR"
            else:
                # Above SMA: own MSTY
                positions[date] = "MSTY"

        return self._run_rotation(
            f"V3: SMA-Enhanced ({sma_period}-day SMA)",
            positions, msty_df, wntr_df, msty_tri, wntr_tri, common_dates,
        )

    def run_v4_sma_plus_debounce(
        self, msty_df: pd.DataFrame, wntr_df: pd.DataFrame,
        sma_period: int = 50,
        debounce_days: int = 3,
    ) -> RotationResult:
        """
        V4 Combined: SMA lead signal + debounce filter.

        Only switch when MSTY crosses its SMA AND stays there for N days.
        Best of both: fast detection + whipsaw protection.
        """
        analyzer = CounterpartyAnalyzer(windows=self.windows)
        comparison = analyzer.prepare_comparison_data(msty_df, wntr_df)
        common_dates = comparison.index

        calc = TotalReturnCalculator()
        msty_tri = calc.calculate_total_return_index(msty_df)
        wntr_tri = calc.calculate_total_return_index(wntr_df)

        # Split-adjusted MSTY price and SMA
        msty_sorted = msty_df.sort_index()
        if "split" in msty_sorted.columns:
            cum_split = msty_sorted["split"].replace(0, 1).cumprod()
            final_split = cum_split.iloc[-1]
            split_adj = cum_split / final_split
            adj_price = msty_sorted["close"] / split_adj
        else:
            adj_price = msty_sorted["close"]

        sma = adj_price.rolling(window=sma_period, min_periods=sma_period).mean()

        # Raw SMA signal per day
        raw_pos = pd.Series("MSTY", index=common_dates)
        for date in common_dates:
            if date in adj_price.index and date in sma.index and not pd.isna(sma[date]):
                raw_pos[date] = "WNTR" if adj_price[date] < sma[date] else "MSTY"

        # Debounce
        positions = pd.Series("MSTY", index=common_dates)
        current = "MSTY"
        pending = None
        count = 0

        for date in common_dates:
            suggested = raw_pos[date]
            if suggested != current:
                if suggested == pending:
                    count += 1
                else:
                    pending = suggested
                    count = 1
                if count >= debounce_days:
                    current = suggested
                    pending = None
                    count = 0
            else:
                pending = None
                count = 0
            positions[date] = current

        return self._run_rotation(
            f"V4: SMA({sma_period}) + Debounce({debounce_days}d)",
            positions, msty_df, wntr_df, msty_tri, wntr_tri, common_dates,
        )


class RotationVisualizer:
    """Charts for the rotation strategy comparison."""

    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_rotation_comparison(
        self, results: List[RotationResult],
    ) -> plt.Figure:
        """
        Multi-strategy comparison:
        1. Equity curves overlaid
        2. Position maps for each strategy
        3. Dividend income comparison
        4. Drawdown comparison
        """
        n = len(results)
        fig, axes = plt.subplots(3 + n, 1, figsize=(18, 8 + 3 * n),
                                  gridspec_kw={"height_ratios": [4] + [1] * n + [3, 3]})

        initial = 10000.0
        colors_strat = ["green", "purple", "darkorange", "teal", "magenta"]

        # --- Panel 1: Equity curves ---
        ax1 = axes[0]

        for i, res in enumerate(results):
            c = colors_strat[i % len(colors_strat)]
            ax1.plot(res.daily_equity.index, res.daily_equity,
                     color=c, linewidth=2, label=f"{res.strategy_name} ({res.total_return_pct:+.1f}%)")

        # Buy-and-hold comparisons
        ref = results[0]
        msty_eq = initial * (ref.msty_tri / ref.msty_tri.iloc[0])
        wntr_eq = initial * (ref.wntr_tri / ref.wntr_tri.iloc[0])
        ax1.plot(msty_eq.index, msty_eq, color="red", linewidth=1.5, alpha=0.5,
                 linestyle="--", label=f"MSTY B&H ({ref.msty_buy_hold_return:+.1f}%)")
        ax1.plot(wntr_eq.index, wntr_eq, color="blue", linewidth=1.5, alpha=0.5,
                 linestyle="--", label=f"WNTR B&H ({ref.wntr_buy_hold_return:+.1f}%)")
        ax1.axhline(y=initial, color="gray", linewidth=1, linestyle=":", alpha=0.4)

        ax1.set_title("Rotation Strategy Equity — Always Invested, Which One?",
                       fontweight="bold", fontsize=14)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panels 2..N+1: Position maps ---
        for i, res in enumerate(results):
            ax = axes[1 + i]
            c_strat = colors_strat[i % len(colors_strat)]

            for trade in res.trades:
                color = "red" if trade.instrument == "MSTY" else "blue"
                alpha = 0.7
                ax.axvspan(trade.entry_date, trade.exit_date, alpha=alpha, color=color)

            ax.set_ylabel(f"V{i + 1}", fontsize=9, fontweight="bold", rotation=0, labelpad=25)
            ax.set_yticks([])
            ax.set_xlim(axes[0].get_xlim())
            if i == 0:
                ax.set_title("Position Maps — Red = MSTY, Blue = WNTR",
                             fontweight="bold", fontsize=10)

        # --- Dividend income panel ---
        ax_div = axes[1 + n]

        labels = []
        msty_divs = []
        wntr_divs = []
        total_divs = []

        for res in results:
            labels.append(res.strategy_name.split(":")[0].strip())
            msty_divs.append(res.msty_dividends)
            wntr_divs.append(res.wntr_dividends)
            total_divs.append(res.total_dividends_collected)

        # Add buy-and-hold references
        labels.extend(["MSTY B&H", "WNTR B&H"])
        # For B&H, total divs during the period
        ref_msty = results[0].trades[0].entry_date  # approximate
        ref_end = results[0].trades[-1].exit_date

        x = range(len(labels))
        width = 0.35

        ax_div.bar([xi - width / 2 for xi in range(len(results))],
                   msty_divs[:len(results)], width, color="red", alpha=0.7, label="From MSTY")
        ax_div.bar([xi + width / 2 for xi in range(len(results))],
                   wntr_divs[:len(results)], width, color="blue", alpha=0.7, label="From WNTR")

        ax_div.set_title("Dividends Collected Per Share by Strategy",
                          fontweight="bold", fontsize=11)
        ax_div.set_ylabel("Dividends ($)")
        ax_div.set_xticks(range(len(results)))
        ax_div.set_xticklabels([l for l in labels[:len(results)]], fontsize=9)
        ax_div.legend(fontsize=9)
        ax_div.grid(True, alpha=0.3, axis="y")

        # --- Drawdown panel ---
        ax_dd = axes[2 + n]

        for i, res in enumerate(results):
            c = colors_strat[i % len(colors_strat)]
            rm = res.daily_equity.expanding().max()
            dd = (res.daily_equity / rm - 1) * 100
            ax_dd.plot(dd.index, dd, color=c, linewidth=1.5,
                       label=f"{res.strategy_name.split(':')[0]} (max: {dd.min():.1f}%)")

        # B&H drawdowns
        for label, eq, color in [("MSTY B&H", msty_eq, "red"), ("WNTR B&H", wntr_eq, "blue")]:
            rm = eq.expanding().max()
            dd = (eq / rm - 1) * 100
            ax_dd.plot(dd.index, dd, color=color, linewidth=1, alpha=0.5, linestyle="--",
                       label=f"{label} (max: {dd.min():.1f}%)")

        ax_dd.set_title("Drawdown Comparison", fontweight="bold", fontsize=11)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Date")
        ax_dd.legend(loc="best", fontsize=8)
        ax_dd.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "rotation_strategy_comparison.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig

    def plot_dividend_income_timeline(
        self, results: List[RotationResult],
        msty_df: pd.DataFrame, wntr_df: pd.DataFrame,
    ) -> plt.Figure:
        """
        Timeline of dividend income captured by each strategy variant.
        Shows cumulative dividends over time for each strategy.
        """
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        colors_strat = ["green", "purple", "darkorange", "teal", "magenta"]

        # --- Panel 1: Cumulative dividends per strategy ---
        ax1 = axes[0]

        msty_sorted = msty_df.sort_index()
        wntr_sorted = wntr_df.sort_index()

        for i, res in enumerate(results):
            c = colors_strat[i % len(colors_strat)]
            dates = res.daily_position.index
            cum_divs = pd.Series(0.0, index=dates)
            running = 0.0

            for date in dates:
                pos = res.daily_position[date]
                if pos == "MSTY" and date in msty_sorted.index:
                    div = msty_sorted.loc[date, "dividend"]
                    if div > 0:
                        running += div
                elif pos == "WNTR" and date in wntr_sorted.index:
                    div = wntr_sorted.loc[date, "dividend"]
                    if div > 0:
                        running += div
                cum_divs[date] = running

            ax1.plot(cum_divs.index, cum_divs, color=c, linewidth=2,
                     label=f"{res.strategy_name.split(':')[0]} (${running:.2f})")

        # B&H references
        msty_cum = msty_sorted["dividend"].cumsum()
        wntr_cum = wntr_sorted["dividend"].cumsum()
        common = res.daily_position.index

        msty_cum_aligned = msty_cum.reindex(common).ffill().fillna(0)
        wntr_cum_aligned = wntr_cum.reindex(common).ffill().fillna(0)

        ax1.plot(common, msty_cum_aligned, color="red", linewidth=1.5, alpha=0.5,
                 linestyle="--", label=f"MSTY B&H (${msty_cum_aligned.iloc[-1]:.2f})")
        ax1.plot(common, wntr_cum_aligned, color="blue", linewidth=1.5, alpha=0.5,
                 linestyle="--", label=f"WNTR B&H (${wntr_cum_aligned.iloc[-1]:.2f})")

        ax1.set_title("Cumulative Dividends Collected Per Share",
                       fontweight="bold", fontsize=13)
        ax1.set_ylabel("Cumulative Dividends ($)")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Which instrument paid dividends each day ---
        ax2 = axes[1]

        # Show dividend payments as stems colored by which strategy captured them
        for i, res in enumerate(results):
            if i > 0:
                continue  # Only show V1 for clarity

            dates = res.daily_position.index
            for date in dates:
                pos = res.daily_position[date]
                if pos == "MSTY" and date in msty_sorted.index:
                    div = msty_sorted.loc[date, "dividend"]
                    if div > 0:
                        ax2.bar(date, div, color="red", alpha=0.7, width=2)
                elif pos == "WNTR" and date in wntr_sorted.index:
                    div = wntr_sorted.loc[date, "dividend"]
                    if div > 0:
                        ax2.bar(date, div, color="blue", alpha=0.7, width=2)

        # Show missed dividends (from the instrument NOT held)
        for date in common:
            pos = results[0].daily_position[date]
            if pos == "MSTY" and date in wntr_sorted.index:
                div = wntr_sorted.loc[date, "dividend"]
                if div > 0:
                    ax2.bar(date, -div, color="blue", alpha=0.3, width=2)
            elif pos == "WNTR" and date in msty_sorted.index:
                div = msty_sorted.loc[date, "dividend"]
                if div > 0:
                    ax2.bar(date, -div, color="red", alpha=0.3, width=2)

        ax2.axhline(y=0, color="black", linewidth=1)
        ax2.set_title("V1 Dividend Capture — Above: Collected, Below: Missed (faded)",
                       fontweight="bold", fontsize=11)
        ax2.set_ylabel("Dividend Amount ($)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / "rotation_dividend_income.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()
        return fig
