"""
V7 Signal Engine for the rotation bot.

Computes the current signal for each pair based on:
  1. Bull ETF price vs 50-day SMA (±5% band)
  2. 3-day debounce
  3. Income/RV ratio filter on bear switches (ratio must be >= 1.5)

This module is stateless — it takes a full price history DataFrame and
returns the current signal state. The bot runner handles persistence.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalState:
    """Complete signal state for one pair on one day."""
    pair_name: str
    date: pd.Timestamp

    # Current position
    position: str          # "BULL" or "BEAR"
    bull_ticker: str
    bear_ticker: str
    holding: str           # The actual ticker being held

    # SMA metrics
    price: float
    sma: float
    pct_sma: float         # (price / sma - 1) * 100
    upper_band: float
    lower_band: float

    # Raw signal (before debounce/filter)
    raw_signal: str        # "BULL", "BEAR", or "HOLD"

    # Debounce state
    pending: Optional[str]  # Pending switch direction, or None
    debounce_count: int     # Days the pending signal has persisted

    # Income/RV metrics
    rv: float              # Realized vol (annualized %)
    income_yield: float    # Income yield (annualized %)
    income_rv_ratio: float # income / rv

    # Filter status
    filter_active: bool    # Is the Inc/RV filter currently blocking?
    filter_blocked: bool   # Was a switch blocked THIS evaluation?

    # Action
    action: str            # "HOLD", "SWITCH_TO_BULL", "SWITCH_TO_BEAR"
    switch_to: Optional[str]  # Ticker to switch to, if action is a switch

    def summary(self) -> str:
        """One-line summary for logging."""
        filt = " [FILTER BLOCKED]" if self.filter_blocked else ""
        pend = f" pending={self.pending}({self.debounce_count}d)" if self.pending else ""
        return (
            f"{self.pair_name}: {self.holding} | "
            f"price=${self.price:.2f} %SMA={self.pct_sma:+.1f}% "
            f"RV={self.rv:.0f}% Inc={self.income_yield:.0f}% "
            f"Inc/RV={self.income_rv_ratio:.2f} | "
            f"signal={self.raw_signal}{pend}{filt} → {self.action}"
        )


def compute_adj_price(df: pd.DataFrame) -> pd.Series:
    """Split-adjusted closing price."""
    df_sorted = df.sort_index()
    if "split" in df_sorted.columns:
        splits = df_sorted["split"].replace(0, 1)
        cum_split = splits.cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        return df_sorted["close"] / split_adj
    return df_sorted["close"]


def compute_rv(adj_price: pd.Series, window: int = 20) -> pd.Series:
    """Realized volatility: annualized std of log returns (%)."""
    log_ret = np.log(adj_price / adj_price.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100


def compute_income_yield(df: pd.DataFrame, rolling_weeks: int = 4) -> pd.Series:
    """Income yield: rolling dividend sum / price, annualized (%)."""
    adj_price = compute_adj_price(df)
    divs = df["dividend"] if "dividend" in df.columns else pd.Series(0, index=df.index)

    if "split" in df.columns:
        splits = df["split"].replace(0, 1)
        cum_split = splits.cumprod()
        final_split = cum_split.iloc[-1]
        split_adj = cum_split / final_split
        adj_divs = divs / split_adj
    else:
        adj_divs = divs

    rolling_days = rolling_weeks * 5
    rolling_div = adj_divs.rolling(rolling_days, min_periods=1).sum()
    return (rolling_div / adj_price) * (52 / rolling_weeks) * 100


def evaluate_pair(
    bull_df: pd.DataFrame,
    pair_name: str,
    bull_ticker: str,
    bear_ticker: str,
    sma_period: int = 50,
    band_pct: float = 5.0,
    debounce_days: int = 3,
    rv_window: int = 20,
    income_weeks: int = 4,
    min_ratio: float = 1.5,
) -> SignalState:
    """
    Run V7 logic over the full history and return the current signal state.

    This replays the entire history to determine:
    - Current position (BULL or BEAR)
    - Pending debounce state
    - Whether a switch is triggered today

    Args:
        bull_df: Bull ETF DataFrame (close, dividend, split)
        pair_name: Display name for the pair
        bull_ticker: Bull ETF ticker
        bear_ticker: Bear ETF ticker
        Remaining args: V7 strategy parameters

    Returns:
        SignalState for the most recent trading day
    """
    adj_price = compute_adj_price(bull_df)
    sma = adj_price.rolling(sma_period, min_periods=sma_period).mean()
    rv = compute_rv(adj_price, rv_window)
    income = compute_income_yield(bull_df, income_weeks)
    ratio = income / rv.replace(0, np.nan)

    # Trim to valid signal dates
    sma_valid = sma.dropna()
    rv_valid = rv.dropna()
    ratio_valid = ratio.dropna()

    if len(sma_valid) == 0 or len(rv_valid) == 0:
        logger.error(f"{pair_name}: Not enough data for SMA/RV")
        # Return a default state
        last = adj_price.index[-1]
        return SignalState(
            pair_name=pair_name, date=last,
            position="BULL", bull_ticker=bull_ticker, bear_ticker=bear_ticker,
            holding=bull_ticker,
            price=adj_price.iloc[-1], sma=np.nan, pct_sma=np.nan,
            upper_band=np.nan, lower_band=np.nan,
            raw_signal="HOLD", pending=None, debounce_count=0,
            rv=np.nan, income_yield=np.nan, income_rv_ratio=np.nan,
            filter_active=False, filter_blocked=False,
            action="HOLD", switch_to=None,
        )

    first_valid = max(sma_valid.index[0], rv_valid.index[0])
    if len(ratio_valid) > 0:
        first_valid = max(first_valid, ratio_valid.index[0])

    dates = adj_price.index[adj_price.index >= first_valid]

    # Determine initial position from SMA
    initial_price = adj_price[dates[0]]
    initial_sma = sma[dates[0]]
    current = "BEAR" if initial_price < initial_sma else "BULL"

    band_mult = band_pct / 100.0
    pending = None
    count = 0
    last_blocked = False

    for date in dates:
        if date not in adj_price.index or date not in sma.index or pd.isna(sma[date]):
            continue

        price = adj_price[date]
        sma_val = sma[date]
        upper = sma_val * (1 + band_mult)
        lower = sma_val * (1 - band_mult)

        # Raw signal
        if price > upper:
            sig = "BULL"
        elif price < lower:
            sig = "BEAR"
        else:
            sig = "HOLD"

        last_blocked = False

        if sig == "HOLD":
            pending = None
            count = 0
        else:
            suggested = sig  # "BULL" or "BEAR"
            if suggested != current:
                if suggested == pending:
                    count += 1
                else:
                    pending = suggested
                    count = 1

                if count >= debounce_days:
                    execute = True

                    # Income/RV filter on bear switches only
                    if suggested == "BEAR" and current == "BULL":
                        if date in ratio.index and not pd.isna(ratio[date]):
                            if ratio[date] < min_ratio:
                                execute = False
                                last_blocked = True
                                pending = None
                                count = 0

                    if execute:
                        current = suggested
                        pending = None
                        count = 0
            else:
                pending = None
                count = 0

    # Build final state from the last date
    last_date = dates[-1]
    price_now = adj_price[last_date]
    sma_now = sma[last_date]
    pct_sma = (price_now / sma_now - 1) * 100
    upper_now = sma_now * (1 + band_mult)
    lower_now = sma_now * (1 - band_mult)
    rv_now = rv[last_date] if last_date in rv.index else np.nan
    inc_now = income[last_date] if last_date in income.index else np.nan
    rat_now = ratio[last_date] if last_date in ratio.index else np.nan

    if price_now > upper_now:
        raw_sig = "BULL"
    elif price_now < lower_now:
        raw_sig = "BEAR"
    else:
        raw_sig = "HOLD"

    filter_active = not pd.isna(rat_now) and rat_now < min_ratio

    holding = bull_ticker if current == "BULL" else bear_ticker

    # Determine action (what changed on the last evaluation)
    # We need to check if the current position changed from what it was
    # the day before. Re-derive by running one step back.
    # Simpler: track the position as of yesterday vs today.
    # For now, the action is computed by the runner comparing
    # today's state with yesterday's persisted state.
    action = "HOLD"
    switch_to = None

    return SignalState(
        pair_name=pair_name, date=last_date,
        position=current, bull_ticker=bull_ticker, bear_ticker=bear_ticker,
        holding=holding,
        price=price_now, sma=sma_now, pct_sma=pct_sma,
        upper_band=upper_now, lower_band=lower_now,
        raw_signal=raw_sig, pending=pending, debounce_count=count,
        rv=rv_now if not pd.isna(rv_now) else 0.0,
        income_yield=inc_now if not pd.isna(inc_now) else 0.0,
        income_rv_ratio=rat_now if not pd.isna(rat_now) else 0.0,
        filter_active=filter_active, filter_blocked=last_blocked,
        action=action, switch_to=switch_to,
    )
