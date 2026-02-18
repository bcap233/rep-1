# Polymarket Arbitrage Bot

## IMPORTANT: Bot Uptime

The trading bot must stay running continuously to collect clean data. **At the start of every session**, run the ensure script before doing anything else:

```bash
./ensure_bot.sh
```

This is idempotent — it ensures cron is running, the cron watchdog entry exists, and the bot is alive. Even between sessions, a cron job runs `watchdog.sh` every 5 minutes to auto-restart the bot if it dies.

Other commands:
- `./watchdog.sh status` — check if bot is alive + last activity
- `./watchdog.sh restart` — stop and restart
- `./watchdog.sh stop` — stop the bot
- `./ensure_bot.sh` — full startup: cron + crontab + bot (run this first)

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
