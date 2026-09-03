"""DRY-RUN-ONLY live executor for the dual-momentum book.

Same shape as alpaca-paper-trader's pairs_signal.py -- compute the live
signal, size it against account equity, and route every prospective
order through the RiskGate -- but with one hard difference: this script
NEVER submits an order. It prints the rebalancing plan and validates it
against the risk gate, then stops. There is no code path that calls
submit_limit_order, on purpose (see DRY_RUN below). Turning it into a
real executor should be an explicit, reviewed change, not a flag someone
flips by accident.

Signal parity: the live signal is computed from the SAME yfinance
adjusted-close series the backtest used, resampled to month-end, via the
same latest_target_weights() function -- so what this would trade is
exactly the rule that was walk-forward tested, not a re-derivation.

The chosen parameters are the ones frozen from the in-sample search in
run_backtest.py (lookback=3, top_k=2). If you re-tune, update them here
too -- they are intentionally hard-coded, not recomputed inline, so the
live decision always uses a model that was actually backtested.
"""
import os
from datetime import date, timedelta

from momentum_data import DEFAULT_UNIVERSE, load_universe, to_month_end
from momentum_strategy import MomentumConfig, latest_target_weights
from risk_gates import RiskGate, RiskLimits

# ---- HARD SAFETY: this executor cannot place orders. ----
DRY_RUN = True  # there is no branch that submits when this is True; see main().

# Frozen from run_backtest.py's in-sample selection. Not recomputed here.
CONFIG = MomentumConfig(lookback=3, top_k=2, skip=1, cost_bps=10.0, absolute_filter=True)

DEFAULT_EQUITY = 10_000.0  # used only if no Alpaca account is reachable


def get_account_equity_and_positions():
    """Read paper-account equity and current positions if Alpaca creds
    are present (a read-only call, no orders). Falls back to a flat
    assumed equity and empty book so the dry run works with no creds at
    all -- the whole point is that it never needs order permissions.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not (api_key and secret_key):
        print("(no ALPACA_API_KEY/SECRET in env -- using assumed equity, empty book)")
        return DEFAULT_EQUITY, {}
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        equity = float(client.get_account().equity)
        positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}
        print(f"(Alpaca paper account: equity=${equity:,.0f}, "
              f"{len(positions)} open positions)")
        return equity, positions
    except Exception as exc:  # network / auth / lib missing -- degrade to dry defaults
        print(f"(could not reach Alpaca: {exc}; using assumed equity, empty book)")
        return DEFAULT_EQUITY, {}


def main():
    print("=== Dual-momentum DRY-RUN executor (no orders will be sent) ===")
    print(f"config: lookback={CONFIG.lookback}m top_k={CONFIG.top_k} "
          f"skip={CONFIG.skip} absolute_filter={CONFIG.absolute_filter}\n")

    # ~3y of daily bars is plenty for a 3-month lookback; matches backtest source.
    start = (date.today() - timedelta(days=3 * 365 + 30)).isoformat()
    daily = load_universe(DEFAULT_UNIVERSE, start=start)
    monthly = to_month_end(daily)
    last_px = daily.iloc[-1]  # latest daily close, for share sizing
    as_of = monthly.index[-1].date()

    weights = latest_target_weights(monthly, CONFIG)
    equity, positions = get_account_equity_and_positions()

    held = weights[weights > 0]
    invested = float(held.sum())
    print(f"signal as of month-end {as_of}: "
          f"holding {list(held.index)} at {invested:.0%} invested "
          f"({1 - invested:.0%} cash)\n")

    # Target shares from equal-weight target notional.
    target_shares = {}
    for sym in DEFAULT_UNIVERSE:
        notional = weights[sym] * equity
        target_shares[sym] = int(notional // last_px[sym]) if weights[sym] > 0 else 0

    # A gate instance purely to VALIDATE the plan -- proves the intended
    # orders would pass the same safety checks the live pairs strategy uses.
    gate = RiskGate(limits=RiskLimits(
        max_position_per_symbol=10_000, max_total_notional=equity * 1.05,
        max_daily_loss=equity * 0.05, max_orders_per_minute=20,
    ))
    gate.positions = {s: q for s, q in positions.items()}

    print(f"{'symbol':>6}  {'target':>7}  {'current':>7}  {'delta':>6}  {'side':>4}  {'gate':>8}")
    any_trade = False
    for sym in DEFAULT_UNIVERSE:
        tgt = target_shares[sym]
        cur = int(positions.get(sym, 0))
        delta = tgt - cur
        if delta == 0:
            continue
        any_trade = True
        side = "BUY" if delta > 0 else "SELL"
        try:
            gate.check_order(sym, side, abs(delta), float(last_px[sym]), now=0.0)
            verdict = "ok"
        except Exception as exc:
            verdict = f"BLOCKED: {exc}"
        print(f"{sym:>6}  {tgt:>7}  {cur:>7}  {delta:>+6}  {side:>4}  {verdict:>8}")

    if not any_trade:
        print("  (current book already matches target -- nothing to trade)")

    print()
    assert DRY_RUN, "DRY_RUN must be True; this script has no reviewed live-order path"
    print("DRY RUN COMPLETE -- no orders submitted. To trade this, build an explicit,")
    print("reviewed executor that routes the deltas above through GatedOrderRouter.")


if __name__ == "__main__":
    main()
