# Polymarket Arbitrage Bot — Trading Methodology & Rules

**Mode:** PAPER (simulated)
**Bankroll:** $5,000 initial → ~$11,400 current
**Active Assets:** BTC only (ETH and SOL paused — thin liquidity, negative PnL)

---

## Strategy 1: Market Maker (Primary — 60% of capital)

**Goal:** Profit from the bid-ask spread on BTC Up/Down short-duration markets, not from predicting direction.

### How It Works
1. Post passive limit bids on **both** the "Up" and "Down" sides of BTC 5m and 15m markets
2. Bid = best_bid + $0.01 (queue priority) capped at fair_value - half_spread - inventory_skew
3. When someone market-sells into our bid, we buy cheap; then post a sell at midpoint or above
4. Round-trip profit = spread captured

### Fair Value Calculation
- **Base:** Order book midpoint (not 50/50 — uses actual market price)
- **Momentum bias:** ±2% weighting from spot momentum consensus (kept low intentionally — MM edge comes from spread, not direction)
- **Formula:** `fair_up = clamp(book_midpoint + momentum_score * 0.02, 0.05, 0.95)`
- **Down side:** `fair_down = 1.0 - fair_up`

### Inventory Management
- Track net inventory per market (positive = long Up, negative = long Down)
- **Skew formula:** `skew = (inventory / max_inventory) * 0.03`
- At max inventory, quotes shift 3c away from the heavy side to encourage offloading
- Max inventory: 250 shares net in either direction

### Sizing
- Base: $100 per side
- **Spread multiplier:** Tighter spreads → bigger size (1.0x at 3c+ spread, up to 1.5x at 1c spread)
- `size = $100 * spread_mult / our_bid_price`

### Built-in Bilateral Arb
- After quoting, checks if `Up_ask + Down_ask < $0.98`
- If yes, buys cheaper side for a risk-free lock (guaranteed $1.00 payout)
- This is an embedded arb on top of the market-making flow

### Filters
| Rule | Value |
|------|-------|
| Durations traded | 5m, 15m only (4h books too thin — 10c+ slippage) |
| Min book spread | 1c (if tighter, can't compete) |
| Min time to expiry | 2 minutes (don't quote dying markets) |
| Max exposure | $4,800 |
| Max positions | 50 |
| Half spread | 2c per side (4c total) |
| Momentum bias weight | 2% (was 5%, reduced after observing 3:1 Down:Up imbalance on bearish days) |

---

## Strategy 2: High Probability Grinder

**Goal:** Buy tokens priced at 85-97% probability; small profit per win, very high win rate.

### Three Modes (Priority Order)

#### Mode 1: Immediate Lock (Risk-Free)
- Scans BTC Up/Down markets for `Up_ask + Down_ask < $0.98`
- Buys both sides simultaneously → guaranteed $1.00 payout
- Net profit = `1.00 - total_cost - fees`
- Size: $50 per lock (riskless, so larger than directional)
- Confidence: 0.99

#### Mode 2: Delayed Lock
- We already hold one side from a previous cycle
- Other side becomes cheap enough → complete the lock
- Matches original leg size for full hedge
- Same profit math as immediate lock

#### Mode 3: Model-Based Directional
- **Probability model:** Random walk with drift=0
  `P(up) = Φ((spot - reference_price) / (σ√T))`
  where Φ = standard normal CDF, σ = per-minute BTC volatility, T = minutes left
- **Volatility estimation:** Rolling 30-point window (~5 min at 10s intervals), blended 50/50 with prior of 0.04%/min
- **Reference price:** Spot price recorded when market window first opens
- **Entry criteria:**
  - Model probability ≥ 75% (strict — rejects coin-flip entries)
  - Model edge ≥ 3% over market implied probability after fees
  - Positive EV per share
- **Sizing:** Edge-scaled from $20 (at 3% edge) up to $100 (at 15%+ edge), capped at 80% of book depth

### Event Market Scanning (Gamma API — Every 5 Minutes)
- Scans all categories: politics, sports, economics, tech, world, culture, crypto
- Buys YES or NO token on whichever side is priced 85-97%
- **Sweet spot:** 88-95% (higher confidence)

### Filters
| Rule | Value |
|------|-------|
| Probability range | 85% - 97% (event markets) |
| Up/Down min probability | 78% directional, 62% with momentum |
| Sweet spot | 88% - 95% |
| Min liquidity | $5,000 |
| Min volume | $10,000 |
| Max spread | 4c |
| Min ask depth | 50 shares (30 for Up/Down) |
| Min hours to expiry | 1 hour |
| Max days to expiry | 90 days |
| Size per trade | $20 base (event), edge-scaled (Up/Down) |
| Max positions | 50 |
| Max exposure | $2,000 |
| Lock max total | $0.98 |
| Lock min profit | $0.005/share |
| Lock size | $50 |
| Fee rate | 2% on winnings |

### Avoid Keywords
`tweet, say, announce, resign, fire, scandal, hack, exploit`

### Favor Keywords
`will, above, below, reach, remain, stay, end, close`

---

## Strategy 3: Bilateral Arbitrage

**Goal:** Buy all sides of a market when total cost < $1.00 for risk-free profit.

### Three Arb Types

#### Type 1: Intra-Market
- Same market: `YES_ask + NO_ask < $1.00`
- Buy both → one MUST resolve to $1.00
- Net profit = `1.00 - YES_ask - NO_ask - fee`
- Min gap after fees: 2c

#### Type 2: Cross-Market
- Two separate but logically complementary markets
- Example: "BTC above $100k" + "BTC below $100k" = guaranteed $1.00
- Uses keyword matching: above/below, over/under, higher/lower, up/down, rise/fall, exceed/drop
- Matches by asset + target price (rounded to nearest $100) + opposite direction
- Min gap after fees: 3c

#### Type 3: Multi-Outcome (Bracket)
- Markets with 3+ outcomes where sum of all best asks < $1.00
- Example: 4 price brackets totaling $0.93 → buy all for 7c profit
- Min gap after fees: 4c

### Filters
| Rule | Value |
|------|-------|
| Min gap (intra) | 2c net of fees |
| Min gap (cross) | 3c net of fees |
| Min gap (multi) | 4c net of fees |
| Fee rate | 2% on winnings |
| Min liquidity per side | $2,000 |
| Min depth per side | 20 shares |
| Max spread per side | 5c |
| Size per leg | $50 |
| Max positions | 20 |
| Max exposure | $2,000 |
| Scan timeout | 45 seconds per scan type |
| Max expiry gap (cross) | 24 hours between paired markets |

---

## Strategy 4: Spot Divergence (Scout Mode — No Direct Execution)

**Goal:** Feed momentum signals to other strategies; detect when spot exchange prices diverge from Polymarket implied probabilities.

### How It Works
1. Fetch composite spot price from 3 exchanges (weighted: Binance 50%, Coinbase 30%, Kraken 20%)
2. Build 5/10/15 minute candles, compute momentum indicators (ROC, VWAP deviation, RSI-14, volume ratio)
3. Multi-timeframe consensus: require 2/3 timeframes confirming same direction
4. Compare fair probability (momentum-derived) vs Polymarket implied probability
5. Signal when edge > 5c

### Momentum Thresholds
| Indicator | Threshold |
|-----------|-----------|
| Momentum (ROC) | 0.3% per timeframe |
| VWAP deviation | 0.2% |
| RSI overbought | 70 |
| RSI oversold | 30 |
| Min confirming timeframes | 2 of 3 |
| Min momentum score | ±0.10 to generate signal |
| Min edge | 5c (probability points) |
| Max spread | 8c |

### Confidence Weighting
| Factor | Weight |
|--------|--------|
| Momentum | 35% |
| VWAP deviation | 25% |
| RSI | 20% |
| Volume | 10% |
| Multi-timeframe agreement | 10% |

### Note
Spot Divergence currently operates as a **scout** — it feeds directional conviction into the Market Maker's momentum bias and the Grinder's Up/Down entries. It does not place standalone trades.

---

## Global Risk Management

### Position Limits
| Parameter | Value |
|-----------|-------|
| Max position per trade | $250 |
| Max total exposure | $8,000 |
| Max open positions | 120 |
| Max MM exposure | $4,800 |
| Max Grinder exposure | $2,000 |
| Max Bilateral exposure | $2,000 |

### Exit Rules
| Rule | Value |
|------|-------|
| Stop loss | $0.18 (18c move against) |
| Take profit | $0.20 (20c move in favor) |
| Max hold time | 120 minutes |
| Loss cooldown | 30 seconds per market after a loss |
| Daily loss limit | $200 (stop all trading for the day) |

### Kelly Criterion Bankroll Management
| Parameter | Value |
|-----------|-------|
| Initial bankroll | $5,000 |
| Kelly fraction | 0.50 (half-Kelly) |
| Min trades before Kelly | 20 |
| Rolling window | 100 trades |
| Min scale | 0.5x base size |
| Max scale | 3.0x base size |
| Drawdown throttle | 15% from peak → force 1.0x scale |

**Half-Kelly rationale:** ~75% of full-Kelly growth with much less variance. At 100+ trades, the rolling win rate and average win/loss ratio feed into the Kelly formula to dynamically adjust position sizing.

---

## Execution

| Parameter | Value |
|-----------|-------|
| Mode | PAPER (simulated) |
| Order type | Limit |
| Limit offset | 1c from fair value (conservative) |
| Poll interval | 10 seconds |
| Market refresh | Every 5 minutes |

---

## Asset Configuration

| Asset | Status | Reason |
|-------|--------|--------|
| **BTC** | ACTIVE | Primary profit driver, deep liquidity, strong Up/Down markets |
| **ETH** | PAUSED | Thin liquidity, negative PnL, -$73 on limited trades |
| **SOL** | PAUSED | Thin liquidity, negative PnL, -$40 on limited trades |

---

## Performance Summary (as of Feb 16, 2026)

| Metric | Value |
|--------|-------|
| Total PnL | +$6,399.94 |
| Starting bankroll | $5,000 |
| Current bankroll | ~$11,400 |
| Total trades | 1,821 (895 closed) |
| Win rate | 56.8% |
| Profit factor | 1.49 |
| Runtime | ~44 hours |
| Best asset | BTC (all profit) |
| Last 100 trades | -$114 (edge compression noted) |

---

## Operational Rules

1. **Bot runs 24/7** in a tmux session (`polybot`) for persistence
2. **Watchdog** (`watchdog.sh`) checks PID + pgrep fallback, restarts if needed — run at start of every session
3. **Crash recovery** (`run_bot.sh`) has exponential backoff (2s → 4s → 8s → ... → 300s cap)
4. **Logs** written to `data/logs/bot_*.log` (one per session)
5. **Trade ledger** at `polymarket/trades.jsonl` (append-only, every trade recorded)
6. **Bankroll state** at `data/bankroll.json` (updated after every closed trade)
7. **Position state** at `polymarket/state.json` (open positions, daily PnL)
