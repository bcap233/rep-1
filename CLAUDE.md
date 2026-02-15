# Polymarket Arbitrage Bot

## IMPORTANT: Bot Uptime

The trading bot must stay running continuously to collect clean data. **At the start of every session**, run the watchdog before doing anything else:

```bash
./watchdog.sh start
```

This is idempotent — safe to call even if the bot is already running. It checks for the process and only starts a new one if needed. The bot runs inside a tmux session (`polybot`) for persistence.

Other watchdog commands:
- `./watchdog.sh status` — check if bot is alive + last activity
- `./watchdog.sh restart` — stop and restart
- `./watchdog.sh stop` — stop the bot

## Architecture

- **run_bot.sh**: Persistent wrapper with crash recovery (exponential backoff)
- **watchdog.sh**: Idempotent launcher that ensures the bot is running in a tmux session
- **polymarket/**: Core bot code (runner.py, strategies, execution engine)
- **data/logs/**: Trading logs (one file per session)
- **data/bankroll.json**: Rolling bankroll state (100-trade window)
- **polymarket/state.json**: Open positions and daily PnL
- **polymarket/trades.jsonl**: Complete trade ledger (append-only)

## Key Files for Analysis

- `polymarket/trades.jsonl` — every trade with timestamp, action, price, size, pnl
- `data/bankroll.json` — current bankroll, peak, cumulative PnL
- `polymarket/state.json` — open positions, daily stats

## Bot Mode

Currently running in **PAPER** mode. All trades are simulated.
