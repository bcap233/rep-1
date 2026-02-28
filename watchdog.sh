#!/usr/bin/env bash
# watchdog.sh — Ensures the Polymarket bot is running.
# Safe to call repeatedly (idempotent). Call this at the start of every session.
#
# Usage:
#   ./watchdog.sh          # Check and start if needed
#   ./watchdog.sh status   # Just check, don't start
#   ./watchdog.sh stop     # Stop the bot
#   ./watchdog.sh restart  # Stop then start

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIDFILE="data/bot.pid"
LOGDIR="data/logs"
mkdir -p "$LOGDIR"

is_bot_running() {
    # Check PID file
    if [ -f "$PIDFILE" ]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    # Fallback: check for the python process directly
    if pgrep -f "python3 -m polymarket --loop" > /dev/null 2>&1; then
        # Update PID file with actual PID
        pgrep -f "python3 -m polymarket --loop" | head -1 > "$PIDFILE"
        return 0
    fi
    # Also check if the run_bot.sh wrapper is alive (it may be in backoff sleep)
    if pgrep -f "bash run_bot.sh" > /dev/null 2>&1; then
        pgrep -f "bash run_bot.sh" | head -1 > "$PIDFILE"
        return 0
    fi
    return 1
}

start_bot() {
    if is_bot_running; then
        local pid
        pid=$(cat "$PIDFILE")
        echo "[watchdog] Bot already running (PID $pid). No action needed."
        return 0
    fi

    echo "[watchdog] Bot not running. Starting via tmux..."

    # Kill any stale tmux session and orphaned processes
    tmux kill-session -t polybot 2>/dev/null || true
    pkill -f "python3 -m polymarket --loop" 2>/dev/null || true
    rm -f "data/run_bot.lock"
    sleep 1

    # Start in a tmux session (survives terminal close)
    tmux new-session -d -s polybot "cd $SCRIPT_DIR && bash run_bot.sh"

    # Wait for python process to appear
    local retries=0
    while [ $retries -lt 15 ]; do
        sleep 1
        if pgrep -f "python3 -m polymarket --loop" > /dev/null 2>&1; then
            pgrep -f "python3 -m polymarket --loop" | head -1 > "$PIDFILE"
            local pid
            pid=$(cat "$PIDFILE")
            echo "[watchdog] Bot started successfully (PID $pid)."

            # Show last few lines of the latest log to confirm
            local latest
            latest=$(ls -t "$LOGDIR"/bot_*.log 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                echo "[watchdog] Latest log: $latest"
                tail -3 "$latest"
            fi
            return 0
        fi
        retries=$((retries + 1))
    done

    echo "[watchdog] ERROR: Bot failed to start within 15s. Check logs."
    return 1
}

stop_bot() {
    echo "[watchdog] Stopping bot..."
    # Kill python process
    if [ -f "$PIDFILE" ]; then
        local pid
        pid=$(cat "$PIDFILE")
        kill "$pid" 2>/dev/null || true
    fi
    pkill -f "python3 -m polymarket --loop" 2>/dev/null || true
    tmux kill-session -t polybot 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "[watchdog] Bot stopped."
}

status() {
    if is_bot_running; then
        local pid
        pid=$(cat "$PIDFILE")
        local uptime_info
        uptime_info=$(ps -o etime= -p "$pid" 2>/dev/null || echo "unknown")
        echo "[watchdog] Bot is RUNNING (PID $pid, uptime: $uptime_info)"

        local latest
        latest=$(ls -t "$LOGDIR"/bot_*.log 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "[watchdog] Latest log: $latest"
            echo "[watchdog] Last activity:"
            tail -5 "$latest"
        fi
        return 0
    else
        echo "[watchdog] Bot is NOT running."
        return 1
    fi
}

case "${1:-start}" in
    start)   start_bot ;;
    stop)    stop_bot ;;
    restart) stop_bot; sleep 2; start_bot ;;
    status)  status ;;
    *)       echo "Usage: $0 {start|stop|restart|status}"; exit 1 ;;
esac
