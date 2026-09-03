# momentum-trend-strategy

A cross-asset **momentum / trend-following** strategy on a basket of
liquid ETFs, built as a deliberate *diversifier* for a portfolio that is
otherwise mostly mean-reversion (pairs stat-arb, event-window fades,
order-flow imbalance reversion). Momentum and mean-reversion are opposite
bets on the autocorrelation of returns, so — whatever the absolute
numbers — having a real, honestly-backtested trend follower alongside
those is the point.

Data is free daily bars via `yfinance` (same pattern as the
`mm-backtester` project). Everything below was actually run, not
illustrative.

## The strategy — standard "dual momentum", nothing exotic

The signal is a textbook formulation so the result is interpretable
against the known momentum literature (Jegadeesh–Titman 1993 relative
strength; Asness–Moskowitz–Pedersen 2013 "Value and Momentum Everywhere"
for the cross-asset version; Antonacci 2014 "Dual Momentum" for the
absolute-momentum overlay). Each month-end:

1. **Formation.** For every asset compute its trailing `L`-month total
   return, ending one month ago (`skip=1`, the standard guard against
   short-term reversal): `mom_i = P_i[t-1] / P_i[t-1-L] - 1`.
2. **Cross-sectional (relative) momentum.** Rank the basket by `mom`,
   take the top `K`.
3. **Absolute momentum overlay.** Of those `K`, hold only the ones whose
   own momentum is still positive; the rest of the `K` equal-weight
   slots go to **cash**. In a broad selloff every asset's momentum turns
   negative and the book moves to cash instead of holding the "best of a
   bad bunch". This is what makes it a trend follower with a downside
   brake rather than an always-long rotation.
4. **Hold** equal-weight (`1/K` per slot) until the next month-end, then
   repeat.

Prices are `auto_adjust`ed (dividends/splits folded in) so this ranks
*total* return — material for the bond and REIT ETFs where the yield is a
big chunk of the return.

### Universe (10 ETFs, documented choice)

`SPY` (US large-cap), `EFA` (developed ex-US), `EEM` (emerging), `IWM`
(US small-cap), `TLT` (long Treasuries), `IEF` (intermediate
Treasuries), `LQD` (IG corporates), `GLD` (gold), `DBC` (broad
commodities), `VNQ` (US REITs).

Chosen to span asset classes rather than pile into one. Cross-asset
momentum is the canonical application in the literature: these move for
different macro reasons and are correlated loosely enough that a
relative-strength rotation has something real to rotate between. A large
single-class universe (say 50 tech names) would just be a leveraged bet
on that class, not a diversifier. All 10 have daily history back past
2007, giving a long walk-forward window.

## Walk-forward discipline

- Month-end history: **2007-01 → 2026-09** (237 months).
- **Train = first 70%** (through 2020-10, 165 months). **Out-of-sample =
  final 30%** (2020-11 → 2026-09, 71 realized months).
- Lookback `L ∈ {3,6,9,12}` and holdings `K ∈ {2,3,4,5}` are tuned on
  train **only**, by net Sharpe. The OOS window is evaluated **exactly
  once** with the parameters frozen.
- **Transaction costs are charged on turnover every rebalance and are
  not in the tunable grid** — a fixed 10 bps of turned-over notional, a
  conservative bid/ask+slippage proxy for these liquid ETFs (Alpaca
  charges no commission, but you still pay the spread). You must never be
  able to tune your costs away.

Selected on train: **`lookback=3`, `top_k=2`** (train Sharpe 0.44).
Note the short 3-month lookback won in-sample rather than the classic
"12-1" — reported as-is; it's a mild flag that the tuning may be fitting
this particular sample's medium-term reversal, which is exactly the kind
of thing the frozen-OOS test exists to check.

## The real out-of-sample result (run once, frozen params)

| metric | strategy (OOS) |
|---|---|
| months | 71 |
| total return | **+93.2%** |
| CAGR | **11.8%** |
| annualized vol | 13.0% |
| **Sharpe** (rf=2%) | **0.77** |
| max drawdown | −21.5% |
| win rate (months) | 66% |
| best / worst month | +10.2% / −6.3% |

### Against passive benchmarks over the *same* OOS window

| | CAGR | Sharpe | max DD |
|---|---|---|---|
| **momentum strategy** | 11.8% | **0.77** | −21.5% |
| equal-weight basket (monthly rebal.) | 8.9% | 0.66 | −19.9% |
| **SPY buy & hold** | **17.0%** | **0.98** | −23.9% |

Monthly-return correlation of the strategy to SPY over OOS: **0.50**.

## Honest assessment

**No artifact flag.** OOS Sharpe 0.77 and a 66% monthly win rate sit
squarely in the plausible range for real momentum on liquid ETFs — the
literature lives around Sharpe 0.4–0.8 net of costs. There is no Sharpe-5
/ 100%-win-rate red flag here; the backtest is using genuine market data
with real costs and a strict train/OOS wall, and the numbers look like
what an honest version of this strategy *should* produce. (The
`run_backtest.py` output prints an explicit warning if it ever sees
Sharpe > 5 or win rate > 95%, precisely so a too-good result gets called
an artifact instead of a win.)

**But read the benchmark table before calling it a win.** Over this
specific OOS window the strategy:

- **beat the equal-weight passive basket** on both return and Sharpe —
  so the momentum *timing* added value over just holding everything; and
- **lost to plain SPY buy-and-hold** on both CAGR (11.8% vs 17.0%) and
  Sharpe (0.77 vs 0.98).

That underperformance vs SPY is not surprising and shouldn't be spun
away: 2020-11 → 2026-09 was a strong, concentrated US-equity bull market,
which is close to the worst regime for a diversified/defensive rotation
that spends time in bonds, gold, commodities and cash. A trend follower
earns its keep in *trending and crisis* markets (2008, 2022's bond+equity
selloff), not in a relentless equity melt-up — and this OOS window is
mostly the latter.

**So what is this actually good for?** The value proposition here is not
"beats the S&P" — it's **diversification**. Correlation to SPY of 0.50,
the ability to sit in cash/bonds/gold when equities break down, and a
return stream driven by a *momentum* mechanism that is structurally the
opposite of the mean-reversion strategies filling the rest of this
portfolio. As a standalone bet to replace an index fund: no. As a
low-correlation sleeve whose bad years (equity bull markets) tend to be
other sleeves' good years: that's a real, defensible role, and that was
the goal of the exercise.

### Deployability

- **Mechanically deployable, low operational risk.** Monthly rebalance,
  ~10 liquid ETFs, no leverage, no shorting, no intraday timing. The
  10 bps cost assumption is conservative for these names, so live costs
  are unlikely to be a nasty surprise — turnover is low (monthly).
- **Caveats before risking money.** (1) One OOS path is one draw; a
  single 71-month window can't distinguish skill from luck, and the
  regime was unusually hostile to the style. (2) The 3-month lookback
  winning in-sample is a soft overfitting signal worth watching. (3) The
  honest expectation is index-like-or-below returns with *different*
  drawdowns, valuable in a portfolio, not on its own.
- **Not deployed.** The live executor here is **dry-run only** (see
  below).

## Live executor — dry-run only, nothing scheduled

`src/live_executor.py` mirrors the structure of the
`alpaca-paper-trader` pairs executor (`pairs_signal.py`) — compute the
live signal, size it against account equity, route every prospective
order through the shared `RiskGate` — with one hard difference: **it
never submits an order.** It prints the rebalancing plan, validates each
intended order against `risk_gates.py`, and stops. There is no code path
that calls `submit_limit_order`. Making it trade for real should be an
explicit, reviewed change, not a flag flipped by accident.

The live signal is computed from the **same** yfinance adjusted-close
series the backtest used, via the same `latest_target_weights()`, using
the **frozen** `lookback=3, top_k=2` — so what it would trade is exactly
the rule that was walk-forward tested, not a re-derivation.

`risk_gates.py` and `alpaca_adapter.py` are reused verbatim from
`alpaca-paper-trader`.

```
$ python3 src/live_executor.py
=== Dual-momentum DRY-RUN executor (no orders will be sent) ===
config: lookback=3m top_k=2 skip=1 absolute_filter=True
signal as of month-end 2026-09-30: holding ['EFA', 'DBC'] at 100% invested (0% cash)
symbol   target  current   delta  side      gate
   EFA       46        0     +46   BUY        ok
   DBC      156        0    +156   BUY        ok
DRY RUN COMPLETE -- no orders submitted.
```

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/run_backtest.py     # download data, walk-forward, print the real OOS result
python3 src/live_executor.py    # dry-run rebalance plan (never trades)
python3 -m pytest tests/ -q     # unit tests for the signal + PnL/metrics math
```

The dry-run executor reads paper-account equity/positions from Alpaca if
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are set (a read-only call); with
no creds it falls back to an assumed $10k book and still runs.

## Structure

- `src/momentum_data.py` — yfinance adjusted daily closes; universe;
  month-end resampling.
- `src/momentum_strategy.py` — the signal (`compute_momentum`,
  `target_weights`), the monthly-rebalance backtest (`run_momentum`), and
  the live-parity helper (`latest_target_weights`).
- `src/metrics.py` — Sharpe / CAGR / drawdown / win-rate on a monthly
  return series.
- `src/run_backtest.py` — the walk-forward driver (tune on 70%, evaluate
  once on 30%, benchmarks + honesty check).
- `src/live_executor.py` — **dry-run-only** executor.
- `src/risk_gates.py`, `src/alpaca_adapter.py` — reused verbatim from
  `alpaca-paper-trader`.
- `tests/test_momentum.py` — unit tests for the signal and PnL/metric
  math.

## Reproducibility note

`yfinance` pulls live data, so re-running later extends the sample and
can shift the frozen-parameter pick and the exact OOS numbers. The
numbers above are from the run on **2026-09** month-end data. The
discipline — tune on the first 70%, evaluate once on the last 30%, costs
fixed — is what's fixed, not the specific decimals.
