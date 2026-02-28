#!/usr/bin/env bash
# ensure_bot.sh — Ensures cron + bot are alive. Idempotent, safe to call anywhere.
# Called from: .profile, CLAUDE.md session start, or manually.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Ensure cron is running (it doesn't survive container restarts)
if ! pgrep -x cron > /dev/null 2>&1; then
    service cron start > /dev/null 2>&1 || cron > /dev/null 2>&1 || true
fi

# 2. Ensure the watchdog crontab entries exist (periodic + reboot)
if ! crontab -l 2>/dev/null | grep -q "watchdog.sh"; then
    (crontab -l 2>/dev/null; echo "*/5 * * * * ${SCRIPT_DIR}/watchdog.sh start >> ${SCRIPT_DIR}/data/logs/cron_watchdog.log 2>&1") | crontab -
fi
if ! crontab -l 2>/dev/null | grep -q "@reboot.*ensure_bot"; then
    (crontab -l 2>/dev/null; echo "@reboot sleep 10 && ${SCRIPT_DIR}/ensure_bot.sh >> ${SCRIPT_DIR}/data/logs/cron_watchdog.log 2>&1") | crontab -
fi

# 3. Start the bot now if it's not running
"${SCRIPT_DIR}/watchdog.sh" start
