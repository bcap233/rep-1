"""
Collapse detection module for covered call ETFs.

Focuses on identifying when price movement away from the 50 SMA
indicates a true reflexive collapse vs normal volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CollapseEvent:
    """Represents a detected collapse event."""
    start_date: datetime
    end_date: Optional[datetime]
    trigger_distance: float          # % below SMA when collapse triggered
    max_distance: float              # Maximum % below SMA during collapse
    max_drawdown: float              # Peak-to-trough drawdown
    duration_days: int
    velocity_at_trigger: float       # Rate of SMA distance change at trigger
    underlying_correlation: float    # Correlation with underlying during event
    recovery_date: Optional[datetime] = None
    was_true_collapse: bool = False  # Confirmed as true collapse vs false signal
    notes: str = ""


@dataclass
class CollapseSignal:
    """Real-time collapse signal."""
    date: datetime
    signal_type: str                 # "warning", "caution", "danger", "collapse"
    distance_from_sma: float
    velocity: float                  # Daily change in SMA distance
    days_below_threshold: int
    current_drawdown: float
    probability_true_collapse: float # Estimated probability this is real
    recommended_action: str


class CollapseDetector:
    """
    Detects covered call reflexive collapse conditions.

    A "reflexive collapse" in covered call ETFs occurs when:
    1. The underlying starts a significant decline
    2. The covered call premium income cannot offset losses
    3. Price breaks below key moving averages
    4. The decline accelerates as the strategy's mechanics work against it

    Key insight: Not every break below the 50 SMA is a collapse.
    We need to distinguish between:
    - Normal volatility / temporary dips
    - True reflexive collapses that continue to accelerate
    """

    def __init__(
        self,
        sma_period: int = 50,
        distance_thresholds: Dict[str, float] = None,
        velocity_thresholds: Dict[str, float] = None,
    ):
        self.sma_period = sma_period
        self.distance_thresholds = distance_thresholds or {
            "warning": -5,
            "caution": -10,
            "danger": -15,
            "collapse": -20,
        }
        self.velocity_thresholds = velocity_thresholds or {
            "accelerating": -2,
            "rapid": -5,
        }

    def prepare_data(
        self,
        etf_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare combined dataframe with all required metrics.

        Args:
            etf_df: MSTY price data
            underlying_df: MSTR price data

        Returns:
            Combined DataFrame with collapse detection metrics
        """
        # Align dates
        common_dates = etf_df.index.intersection(underlying_df.index)

        df = pd.DataFrame(index=common_dates)
        df["price"] = etf_df.loc[common_dates, "close"]
        df["underlying_price"] = underlying_df.loc[common_dates, "close"]

        # Calculate 50 SMA
        df["sma_50"] = df["price"].rolling(window=self.sma_period).mean()
        df["underlying_sma_50"] = df["underlying_price"].rolling(window=self.sma_period).mean()

        # Distance from SMA (percentage)
        df["sma_distance_pct"] = ((df["price"] - df["sma_50"]) / df["sma_50"]) * 100
        df["underlying_sma_distance_pct"] = (
            (df["underlying_price"] - df["underlying_sma_50"]) / df["underlying_sma_50"]
        ) * 100

        # Velocity: daily change in SMA distance
        df["sma_distance_velocity"] = df["sma_distance_pct"].diff()

        # 3-day velocity (smoothed)
        df["sma_distance_velocity_3d"] = df["sma_distance_pct"].diff(3) / 3

        # Drawdown from rolling peak
        df["rolling_peak"] = df["price"].expanding().max()
        df["drawdown_pct"] = ((df["price"] - df["rolling_peak"]) / df["rolling_peak"]) * 100

        # Rolling correlation with underlying (20-day)
        etf_returns = df["price"].pct_change()
        underlying_returns = df["underlying_price"].pct_change()
        df["correlation_20d"] = etf_returns.rolling(20).corr(underlying_returns)

        # Days below each threshold
        for level, threshold in self.distance_thresholds.items():
            below_col = f"below_{level}"
            df[below_col] = df["sma_distance_pct"] < threshold

            # Consecutive days below
            df[f"consecutive_days_{level}"] = (
                df[below_col]
                .groupby((~df[below_col]).cumsum())
                .cumsum()
            )

        # Underlying relative strength
        df["relative_strength"] = (
            df["price"].pct_change(5) - df["underlying_price"].pct_change(5)
        ) * 100

        return df

    def calculate_collapse_probability(
        self,
        row: pd.Series,
        historical_stats: Dict = None,
    ) -> float:
        """
        Calculate probability that current conditions indicate a true collapse.

        Factors considered:
        1. Distance below SMA (further = higher probability)
        2. Velocity of decline (faster = higher probability)
        3. Consecutive days below threshold
        4. Current drawdown magnitude
        5. Underlying behavior (is MSTR also collapsing?)

        Returns:
            Probability between 0 and 1
        """
        probability = 0.0

        distance = row.get("sma_distance_pct", 0)
        velocity = row.get("sma_distance_velocity_3d", 0)
        drawdown = row.get("drawdown_pct", 0)
        consec_danger = row.get("consecutive_days_danger", 0)
        underlying_dist = row.get("underlying_sma_distance_pct", 0)

        # Factor 1: Distance from SMA (0-30 points)
        if distance < -20:
            probability += 30
        elif distance < -15:
            probability += 25
        elif distance < -10:
            probability += 15
        elif distance < -5:
            probability += 5

        # Factor 2: Velocity (0-25 points)
        if velocity < -5:
            probability += 25
        elif velocity < -3:
            probability += 20
        elif velocity < -2:
            probability += 15
        elif velocity < -1:
            probability += 5

        # Factor 3: Consecutive days in danger zone (0-20 points)
        if consec_danger >= 5:
            probability += 20
        elif consec_danger >= 3:
            probability += 15
        elif consec_danger >= 2:
            probability += 10
        elif consec_danger >= 1:
            probability += 5

        # Factor 4: Drawdown magnitude (0-15 points)
        if drawdown < -25:
            probability += 15
        elif drawdown < -20:
            probability += 12
        elif drawdown < -15:
            probability += 8
        elif drawdown < -10:
            probability += 4

        # Factor 5: Underlying also collapsing (0-10 points)
        if underlying_dist < -15:
            probability += 10
        elif underlying_dist < -10:
            probability += 7
        elif underlying_dist < -5:
            probability += 3

        return min(probability / 100, 1.0)

    def get_current_signal(
        self,
        df: pd.DataFrame,
    ) -> CollapseSignal:
        """
        Get the current collapse signal based on latest data.

        Args:
            df: Prepared dataframe from prepare_data()

        Returns:
            CollapseSignal with current status
        """
        if df.empty:
            return None

        latest = df.iloc[-1]
        date = df.index[-1]

        distance = latest["sma_distance_pct"]
        velocity = latest["sma_distance_velocity_3d"]
        drawdown = latest["drawdown_pct"]

        # Determine signal type
        if distance < self.distance_thresholds["collapse"]:
            signal_type = "collapse"
        elif distance < self.distance_thresholds["danger"]:
            signal_type = "danger"
        elif distance < self.distance_thresholds["caution"]:
            signal_type = "caution"
        elif distance < self.distance_thresholds["warning"]:
            signal_type = "warning"
        else:
            signal_type = "normal"

        # Get consecutive days
        days_below = 0
        for level in ["collapse", "danger", "caution", "warning"]:
            col = f"consecutive_days_{level}"
            if col in latest.index and latest[col] > 0:
                days_below = int(latest[col])
                break

        probability = self.calculate_collapse_probability(latest)

        # Recommended action
        if probability >= 0.7:
            action = "HIGH ALERT: Strong collapse indicators. Consider defensive positioning."
        elif probability >= 0.5:
            action = "ELEVATED RISK: Monitor closely. Prepare exit strategy."
        elif probability >= 0.3:
            action = "CAUTION: Below SMA but not confirmed collapse. Watch velocity."
        else:
            action = "MONITOR: Normal conditions or early warning only."

        return CollapseSignal(
            date=date,
            signal_type=signal_type,
            distance_from_sma=distance,
            velocity=velocity if not pd.isna(velocity) else 0,
            days_below_threshold=days_below,
            current_drawdown=drawdown,
            probability_true_collapse=probability,
            recommended_action=action,
        )

    def identify_historical_collapses(
        self,
        df: pd.DataFrame,
        min_drawdown: float = -15,
        min_duration: int = 3,
    ) -> List[CollapseEvent]:
        """
        Identify historical collapse events for backtesting.

        Args:
            df: Prepared dataframe
            min_drawdown: Minimum drawdown to qualify as collapse
            min_duration: Minimum days below danger threshold

        Returns:
            List of CollapseEvent objects
        """
        collapses = []
        in_collapse = False
        current = None

        for i, (date, row) in enumerate(df.iterrows()):
            distance = row["sma_distance_pct"]
            drawdown = row["drawdown_pct"]
            velocity = row["sma_distance_velocity_3d"]
            correlation = row["correlation_20d"]

            # Check for collapse entry
            if not in_collapse:
                # Enter collapse when below danger threshold
                if distance < self.distance_thresholds["danger"]:
                    in_collapse = True
                    current = {
                        "start_date": date,
                        "trigger_distance": distance,
                        "max_distance": distance,
                        "max_drawdown": drawdown,
                        "velocity_at_trigger": velocity if not pd.isna(velocity) else 0,
                        "correlations": [correlation] if not pd.isna(correlation) else [],
                    }
            else:
                # Update collapse metrics
                if distance < current["max_distance"]:
                    current["max_distance"] = distance
                if drawdown < current["max_drawdown"]:
                    current["max_drawdown"] = drawdown
                if not pd.isna(correlation):
                    current["correlations"].append(correlation)

                # Check for collapse exit (price returns above warning level)
                if distance > self.distance_thresholds["warning"]:
                    in_collapse = False
                    duration = (date - current["start_date"]).days

                    # Only record if meets minimum criteria
                    if (current["max_drawdown"] <= min_drawdown and
                        duration >= min_duration):

                        avg_corr = np.mean(current["correlations"]) if current["correlations"] else 0

                        # Determine if this was a "true" collapse
                        # (deep drawdown + extended duration + high velocity)
                        was_true = (
                            current["max_drawdown"] <= -20 and
                            duration >= 5 and
                            current["velocity_at_trigger"] < -2
                        )

                        collapses.append(CollapseEvent(
                            start_date=current["start_date"],
                            end_date=date,
                            trigger_distance=current["trigger_distance"],
                            max_distance=current["max_distance"],
                            max_drawdown=current["max_drawdown"],
                            duration_days=duration,
                            velocity_at_trigger=current["velocity_at_trigger"],
                            underlying_correlation=avg_corr,
                            recovery_date=date,
                            was_true_collapse=was_true,
                        ))

        # Handle ongoing collapse
        if in_collapse and current:
            duration = (df.index[-1] - current["start_date"]).days
            avg_corr = np.mean(current["correlations"]) if current["correlations"] else 0

            collapses.append(CollapseEvent(
                start_date=current["start_date"],
                end_date=None,  # Still ongoing
                trigger_distance=current["trigger_distance"],
                max_distance=current["max_distance"],
                max_drawdown=current["max_drawdown"],
                duration_days=duration,
                velocity_at_trigger=current["velocity_at_trigger"],
                underlying_correlation=avg_corr,
                was_true_collapse=False,  # Can't confirm yet
                notes="ONGOING - Collapse still in progress"
            ))

        return collapses

    def analyze_threshold_effectiveness(
        self,
        df: pd.DataFrame,
        forward_days: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Analyze how predictive different SMA distance thresholds are.

        For each threshold level, calculate:
        - How often does crossing it lead to further decline?
        - What's the average forward return after crossing?
        - What's the false positive rate (crosses then recovers)?

        Args:
            df: Prepared dataframe
            forward_days: Forward periods to analyze

        Returns:
            DataFrame with threshold effectiveness metrics
        """
        results = []

        for level, threshold in self.distance_thresholds.items():
            # Find first day crossing below threshold (from above)
            crossed_below = (
                (df["sma_distance_pct"].shift(1) >= threshold) &
                (df["sma_distance_pct"] < threshold)
            )

            cross_dates = df[crossed_below].index

            for fwd in forward_days:
                forward_returns = []
                continued_decline = 0
                recovered = 0

                for cross_date in cross_dates:
                    # Get price at crossing
                    cross_idx = df.index.get_loc(cross_date)

                    if cross_idx + fwd < len(df):
                        cross_price = df.iloc[cross_idx]["price"]
                        future_price = df.iloc[cross_idx + fwd]["price"]
                        fwd_return = (future_price / cross_price - 1) * 100
                        forward_returns.append(fwd_return)

                        if fwd_return < 0:
                            continued_decline += 1
                        else:
                            recovered += 1

                if forward_returns:
                    results.append({
                        "threshold_level": level,
                        "threshold_pct": threshold,
                        "forward_days": fwd,
                        "num_crosses": len(forward_returns),
                        "avg_forward_return": np.mean(forward_returns),
                        "median_forward_return": np.median(forward_returns),
                        "worst_forward_return": np.min(forward_returns),
                        "best_forward_return": np.max(forward_returns),
                        "pct_continued_decline": (continued_decline / len(forward_returns)) * 100,
                        "pct_recovered": (recovered / len(forward_returns)) * 100,
                    })

        return pd.DataFrame(results)

    def find_optimal_threshold(
        self,
        df: pd.DataFrame,
        target_accuracy: float = 0.7,
    ) -> Dict:
        """
        Find the optimal SMA distance threshold that best predicts true collapses.

        We want to find the threshold where:
        - High probability of continued decline (low false positives)
        - Triggers early enough to be useful (not too deep into collapse)

        Args:
            df: Prepared dataframe
            target_accuracy: Target % of crosses that lead to further decline

        Returns:
            Dict with optimal threshold analysis
        """
        test_thresholds = range(-3, -26, -1)  # Test -3% to -25%
        results = []

        for threshold in test_thresholds:
            # Find crosses below this threshold
            crossed_below = (
                (df["sma_distance_pct"].shift(1) >= threshold) &
                (df["sma_distance_pct"] < threshold)
            )

            cross_dates = df[crossed_below].index

            if len(cross_dates) < 2:
                continue

            # Analyze 10-day forward returns
            continued = 0
            avg_further_decline = []

            for cross_date in cross_dates:
                cross_idx = df.index.get_loc(cross_date)

                if cross_idx + 10 < len(df):
                    cross_price = df.iloc[cross_idx]["price"]

                    # Check if price went lower in next 10 days
                    future_prices = df.iloc[cross_idx:cross_idx + 11]["price"]
                    min_future = future_prices.min()

                    if min_future < cross_price:
                        continued += 1
                        decline = (min_future / cross_price - 1) * 100
                        avg_further_decline.append(decline)

            if cross_dates.size > 0:
                accuracy = continued / len(cross_dates)
                avg_decline = np.mean(avg_further_decline) if avg_further_decline else 0

                results.append({
                    "threshold": threshold,
                    "num_signals": len(cross_dates),
                    "accuracy": accuracy,
                    "avg_further_decline": avg_decline,
                })

        results_df = pd.DataFrame(results)

        # Find threshold that meets target accuracy with most signals
        meets_target = results_df[results_df["accuracy"] >= target_accuracy]

        if meets_target.empty:
            # Take the highest accuracy available
            optimal = results_df.loc[results_df["accuracy"].idxmax()]
        else:
            # Take the least negative (earliest warning) that meets target
            optimal = meets_target.loc[meets_target["threshold"].idxmax()]

        return {
            "optimal_threshold": optimal["threshold"],
            "accuracy": optimal["accuracy"],
            "num_signals": optimal["num_signals"],
            "avg_further_decline": optimal["avg_further_decline"],
            "all_results": results_df,
        }
