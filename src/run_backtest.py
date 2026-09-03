"""Walk-forward backtest for the dual-momentum ETF strategy.

Discipline (identical in spirit to mm-backtester's run_pairs_backtest.py):
parameters (lookback L, number of holdings K) are tuned ONLY on the
first ~70% of the month-end history; the final ~30% is evaluated exactly
once, with those parameters frozen, and reported honestly whatever it
shows. Transaction costs are charged on every rebalance and are not part
of the tunable grid.

The whole backtest is run on the full price history for each candidate
config, then the resulting monthly-return series is sliced by date into
train / out-of-sample. Warming up a config's signal on in-sample prices
before the OOS window is NOT look-ahead -- at the real OOS start you
genuinely have all prior prices; the only thing forbidden across the
boundary is choosing parameters with knowledge of OOS returns, and that
is exactly what the frozen-params rule prevents.
"""
import numpy as np
import pandas as pd

from momentum_data import DEFAULT_UNIVERSE, load_universe, to_month_end
from momentum_strategy import MomentumConfig, run_momentum
from metrics import summarize

RF_ANNUAL = 0.02  # flat risk-free proxy for Sharpe; ~avg short T-bill over the sample

LOOKBACKS = [3, 6, 9, 12]   # months
TOP_KS = [2, 3, 4, 5]       # number of equal-weight slots
COST_BPS = 10.0             # fixed, not tuned


def slice_by_date(series: pd.Series, lo=None, hi=None) -> pd.Series:
    out = series
    if lo is not None:
        out = out[out.index > lo]
    if hi is not None:
        out = out[out.index <= hi]
    return out


def benchmark_returns(monthly_prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Passive baselines to beat: monthly-rebalanced equal-weight of the
    whole basket, and SPY buy-and-hold. Both are monthly return series.
    """
    asset_ret = monthly_prices.pct_change().dropna(how="any")
    eq_weight = asset_ret.mean(axis=1)
    eq_weight.name = "equal_weight"
    spy = asset_ret["SPY"].copy()
    spy.name = "SPY_buy_hold"
    return eq_weight, spy


def main():
    print("=== Downloading universe (yfinance, adjusted daily closes) ===")
    daily = load_universe(DEFAULT_UNIVERSE)
    monthly = to_month_end(daily)
    print(f"universe: {', '.join(DEFAULT_UNIVERSE)}")
    print(f"month-end rows: {len(monthly)}  "
          f"({monthly.index[0].date()} .. {monthly.index[-1].date()})")

    # 70/30 split on the month-end calendar.
    split_i = int(len(monthly) * 0.70)
    split_date = monthly.index[split_i]
    print(f"train/OOS split at {split_date.date()} "
          f"(train {split_i} months, OOS {len(monthly) - split_i} months)\n")

    print("=== In-sample parameter search (first 70%, by net Sharpe) ===")
    print(f"{'L':>3} {'K':>3}  {'months':>6}  {'cagr':>7}  {'sharpe':>7}  {'maxDD':>7}")
    best_cfg, best_sharpe, best_train = None, -np.inf, None
    for L in LOOKBACKS:
        for K in TOP_KS:
            cfg = MomentumConfig(lookback=L, top_k=K, cost_bps=COST_BPS)
            res = run_momentum(monthly, cfg)
            train_ret = slice_by_date(res.monthly_returns, hi=split_date)
            m = summarize(train_ret, rf_annual=RF_ANNUAL)
            if m.get("n_months", 0) < 12:
                continue
            print(f"{L:>3} {K:>3}  {m['n_months']:>6}  {m['cagr']:>7.2%}  "
                  f"{m['sharpe']:>7.2f}  {m['max_drawdown']:>7.2%}")
            if m["sharpe"] > best_sharpe:
                best_sharpe, best_cfg, best_train = m["sharpe"], cfg, m

    print(f"\nSelected on train: lookback={best_cfg.lookback}, top_k={best_cfg.top_k}, "
          f"cost_bps={best_cfg.cost_bps}  (train Sharpe {best_train['sharpe']:.2f})")

    print("\n=== Out-of-sample evaluation (final 30%, params frozen, run once) ===")
    res = run_momentum(monthly, best_cfg)
    oos_ret = slice_by_date(res.monthly_returns, lo=split_date)
    m = summarize(oos_ret, rf_annual=RF_ANNUAL)
    for k, v in m.items():
        print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}")

    print("\n=== Passive benchmarks over the SAME out-of-sample window ===")
    eq_bench, spy_bench = benchmark_returns(monthly)
    oos_spy = slice_by_date(spy_bench, lo=split_date)
    for name, series in [("equal-weight basket", eq_bench), ("SPY buy & hold", spy_bench)]:
        bm = summarize(slice_by_date(series, lo=split_date), rf_annual=RF_ANNUAL)
        print(f"  {name:22s} cagr={bm['cagr']:>7.2%}  sharpe={bm['sharpe']:>6.2f}  "
              f"maxDD={bm['max_drawdown']:>7.2%}")

    # The diversification case: how correlated are the strategy's monthly
    # returns to just holding the S&P? Low correlation is the actual point
    # of adding a trend follower to a mean-reversion-heavy book.
    aligned = pd.concat([oos_ret, oos_spy], axis=1, join="inner").dropna()
    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    print(f"  OOS monthly-return correlation strategy vs SPY: {corr:.2f}")

    print("\n=== Honesty check ===")
    if m["sharpe"] > 5 or m["win_rate"] > 0.95:
        print("  ** SUSPICIOUS: Sharpe>5 or win-rate>95% out-of-sample. For a monthly")
        print("     trend strategy on real ETF data this is almost certainly an artifact")
        print("     (bug / look-ahead / survivorship), NOT a real edge. Do not trust it.")
    else:
        print(f"  OOS Sharpe {m['sharpe']:.2f}, win-rate {m['win_rate']:.0%}: in the plausible")
        print("     range for real momentum on liquid ETFs (the literature lives around")
        print("     Sharpe 0.4-0.8 net of costs). No artifact flag raised.")


if __name__ == "__main__":
    main()
