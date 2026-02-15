#!/usr/bin/env bash
# run_bot.sh — Persistent wrapper for the Polymarket arbitrage bot.
# Auto-restarts on crash with exponential backoff.
# Usage:
#   ./run_bot.sh              # Run all strategies in paper mode (default)
#   ./run_bot.sh --live       # Run in live mode
#   ./run_bot.sh --strategy market_maker  # Single strategy
#
# To run in background (survives terminal close):
#   tmux new-session -d -s bot './run_bot.sh'
#   tmux attach -t bot        # Re-attach to see output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/bot_$(date +%Y%m%d_%H%M%S).log"
LATEST_LINK="$LOG_DIR/latest.log"

# Backoff settings
MAX_BACKOFF=300   # 5 minutes max wait between restarts
INITIAL_BACKOFF=5 # 5 seconds initial wait
backoff=$INITIAL_BACKOFF

# Minimum uptime (seconds) before resetting backoff — if the bot ran for
# this long before crashing it was probably a transient error, not a boot loop.
MIN_UPTIME_RESET=120

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cleanup() {
    log "Wrapper script shutting down."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Symlink latest log
ln -sf "$(basename "$LOG_FILE")" "$LATEST_LINK"

log "=== Polymarket Bot Wrapper Starting ==="
log "Log file: $LOG_FILE"
log "Bot args: $*"
log "Python: $(python3 --version 2>&1)"

while true; do
    start_time=$(date +%s)
    log "Starting bot (backoff=${backoff}s)..."

    # Run the bot — all output goes to log and stdout
    set +e
    python3 -m polymarket --loop --quiet "$@" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    set -e

    end_time=$(date +%s)
    uptime=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        log "Bot exited cleanly (code 0). Not restarting."
        break
    fi

    log "Bot crashed (exit code $exit_code) after ${uptime}s uptime."

    # Reset backoff if bot ran long enough (transient crash, not boot loop)
    if [ $uptime -ge $MIN_UPTIME_RESET ]; then
        backoff=$INITIAL_BACKOFF
        log "Uptime > ${MIN_UPTIME_RESET}s — resetting backoff to ${backoff}s."
    fi

    log "Restarting in ${backoff}s..."
    sleep "$backoff"

    # Exponential backoff (capped)
    backoff=$((backoff * 2))
    if [ $backoff -gt $MAX_BACKOFF ]; then
        backoff=$MAX_BACKOFF
    fi
done

log "=== Wrapper exiting ==="
