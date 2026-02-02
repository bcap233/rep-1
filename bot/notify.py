"""
Notification system for the V7 rotation bot.

Sends alerts via webhook (Discord/Slack compatible) when:
  - A position switch is triggered
  - The Income/RV filter blocks a switch
  - Daily status summary
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

from .signal import SignalState

logger = logging.getLogger(__name__)


def send_webhook(url: str, message: str, title: Optional[str] = None) -> bool:
    """
    Send a message to a webhook URL (Discord or Slack compatible).
    Returns True on success.
    """
    if not url:
        return False

    # Try Discord format first, fall back to Slack format
    if "discord.com" in url:
        payload = {
            "embeds": [{
                "title": title or "V7 Rotation Bot",
                "description": message,
                "color": 3447003,  # Blue
            }]
        }
    else:
        # Slack / generic webhook format
        text = f"*{title}*\n{message}" if title else message
        payload = {"text": text}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status < 300:
                logger.info(f"Webhook sent successfully")
                return True
            else:
                logger.warning(f"Webhook returned status {resp.status}")
                return False
    except urllib.error.URLError as e:
        logger.error(f"Webhook failed: {e}")
        return False


def format_switch_alert(
    old_holding: str,
    new_state: SignalState,
) -> str:
    """Format a position switch alert message."""
    lines = [
        f"**{new_state.pair_name}: SWITCH {old_holding} → {new_state.holding}**",
        "",
        f"Price: ${new_state.price:.2f}  |  SMA: ${new_state.sma:.2f}  |  %SMA: {new_state.pct_sma:+.1f}%",
        f"RV: {new_state.rv:.0f}%  |  Income: {new_state.income_yield:.0f}%  |  Inc/RV: {new_state.income_rv_ratio:.2f}",
        f"Band: ${new_state.lower_band:.2f} — ${new_state.upper_band:.2f}",
    ]
    return "\n".join(lines)


def format_filter_alert(new_state: SignalState) -> str:
    """Format a filter-blocked alert."""
    lines = [
        f"**{new_state.pair_name}: BEAR SWITCH BLOCKED by Income/RV filter**",
        "",
        f"Signal says switch to {new_state.bear_ticker}, but Inc/RV = {new_state.income_rv_ratio:.2f} (< 1.5)",
        f"Staying in {new_state.holding}. Vol spike detected — waiting for premium engine to heal.",
        "",
        f"Price: ${new_state.price:.2f}  |  %SMA: {new_state.pct_sma:+.1f}%  |  RV: {new_state.rv:.0f}%  |  Income: {new_state.income_yield:.0f}%",
    ]
    return "\n".join(lines)


def format_daily_summary(states: list[SignalState]) -> str:
    """Format the daily status summary for all pairs."""
    lines = [
        "**V7 Daily Signal Summary**",
        "",
        f"{'Pair':<12} {'Hold':>6} {'%SMA':>7} {'RV':>5} {'Inc':>5} {'I/RV':>5} {'Signal':>6} {'Pend':>8} {'Filter':>7}",
        "-" * 70,
    ]

    for s in states:
        pend_str = f"{s.pending}({s.debounce_count}d)" if s.pending else "—"
        filt_str = "ACTIVE" if s.filter_active else "—"
        lines.append(
            f"{s.pair_name:<12} {s.holding:>6} {s.pct_sma:>+6.1f}% "
            f"{s.rv:>4.0f}% {s.income_yield:>4.0f}% {s.income_rv_ratio:>5.2f} "
            f"{s.raw_signal:>6} {pend_str:>8} {filt_str:>7}"
        )

    return "\n".join(lines)
