"""Real (paper) order-submitting executor for the dual-momentum book --
the explicit, reviewed live path that live_executor.py's own comments
say should exist separately rather than be a flag flip.

This is for long-horizon validation: the backtest showed a real,
literature-consistent OOS Sharpe (0.77), but a backtest Sharpe is not a
live Sharpe. The only way to know if this holds up is to actually run
it on paper for months and compare. Runs monthly (the strategy's own
rebalance frequency -- running more often would just re-check the same
month-end signal, same reasoning as pairs_signal.py's daily-not-hourly
cadence).

Signal parity with the backtest is preserved exactly: same
latest_target_weights() function, same frozen config (lookback=3,
top_k=2, from run_backtest.py's in-sample selection), same universe.
"""
import os
from datetime import date, timedelta

from momentum_data import DEFAULT_UNIVERSE, load_universe, to_month_end
from momentum_strategy import MomentumConfig, latest_target_weights
from risk_gates import RiskGate, RiskLimits
from alpaca_adapter import GatedOrderRouter

CONFIG = MomentumConfig(lookback=3, top_k=2, skip=1, cost_bps=10.0, absolute_filter=True)

# Real capital cap for this validation run -- small on purpose, this is
# about measuring whether the edge is real, not deploying meaningful size.
VALIDATION_EQUITY_CAP = 2_000.0


def main():
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    start = (date.today() - timedelta(days=3 * 365 + 30)).isoformat()
    daily = load_universe(DEFAULT_UNIVERSE, start=start)
    monthly = to_month_end(daily)
    last_px = daily.iloc[-1]
    as_of = monthly.index[-1].date()

    weights = latest_target_weights(monthly, CONFIG)

    gate = RiskGate(limits=RiskLimits(
        max_position_per_symbol=100, max_total_notional=VALIDATION_EQUITY_CAP * 1.1,
        max_daily_loss=VALIDATION_EQUITY_CAP * 0.10, max_orders_per_minute=10,
    ))
    router = GatedOrderRouter(gate, api_key, secret_key, paper=True)
    router.warmup()  # see alpaca-paper-trader's pairs_signal.py fix -- avoid cold TLS on the first real order

    positions = {p.symbol: float(p.qty) for p in router.client.get_all_positions()}
    equity = min(float(router.client.get_account().equity), VALIDATION_EQUITY_CAP)

    held = weights[weights > 0]
    print(f"=== momentum rebalance {as_of} === holding {list(held.index)}, "
          f"equity capped at ${equity:,.0f}")

    target_shares = {}
    for sym in DEFAULT_UNIVERSE:
        notional = weights[sym] * equity
        target_shares[sym] = int(notional // last_px[sym]) if weights[sym] > 0 else 0

    any_order = False
    for sym in DEFAULT_UNIVERSE:
        tgt = target_shares[sym]
        cur = int(positions.get(sym, 0))
        delta = tgt - cur
        if delta == 0:
            continue
        any_order = True
        side = "BUY" if delta > 0 else "SELL"
        # round to valid cent increment -- raw float pricing (e.g.
        # 107.05999755859375) trips Alpaca's sub-penny rejection (42210000)
        price = round(float(last_px[sym]), 2)
        try:
            order = router.submit_limit_order(sym, side, abs(delta), price)
            print(f"  {side} {abs(delta)} {sym} @ {price:.2f} -> order {order.id}")
        except Exception as exc:
            print(f"  {side} {abs(delta)} {sym} @ {price:.2f} -> BLOCKED: {exc}")

    if not any_order:
        print("  book already matches target -- nothing to trade this month")


if __name__ == "__main__":
    main()
