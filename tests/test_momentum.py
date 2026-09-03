import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from momentum_strategy import (
    MomentumConfig,
    compute_momentum,
    target_weights,
    run_momentum,
    latest_target_weights,
)
from metrics import summarize


def _month_index(n):
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def test_compute_momentum_recovers_known_return():
    # One asset that rises exactly 10% each month. With skip=1, lookback=2
    # the momentum at row t is P[t-1]/P[t-3]-1 = 1.1^2 - 1 = 0.21.
    px = pd.DataFrame({"A": [100 * 1.1 ** i for i in range(6)]}, index=_month_index(6))
    mom = compute_momentum(px, lookback=2, skip=1)
    assert np.isnan(mom["A"].iloc[2])          # not enough history yet
    assert abs(mom["A"].iloc[3] - (1.1 ** 2 - 1)) < 1e-9


def test_target_weights_equal_weights_top_k():
    mom = pd.Series({"A": 0.30, "B": 0.20, "C": 0.10, "D": -0.05})
    w = target_weights(mom, top_k=2, absolute_filter=True)
    assert w["A"] == 0.5 and w["B"] == 0.5
    assert w["C"] == 0.0 and w["D"] == 0.0


def test_absolute_filter_sends_negative_momentum_to_cash():
    # Top 2 by rank are A and B, but B's momentum is negative -> only A
    # is held; B's slot goes to cash, so the book is only 50% invested.
    mom = pd.Series({"A": 0.30, "B": -0.10, "C": -0.20})
    w = target_weights(mom, top_k=2, absolute_filter=True)
    assert w["A"] == 0.5
    assert w["B"] == 0.0 and w["C"] == 0.0
    assert w.sum() == 0.5  # deliberately under-invested, not "best of a bad bunch"


def test_absolute_filter_off_holds_full_book():
    mom = pd.Series({"A": 0.30, "B": -0.10, "C": -0.20})
    w = target_weights(mom, top_k=2, absolute_filter=False)
    assert w["A"] == 0.5 and w["B"] == 0.5
    assert abs(w.sum() - 1.0) < 1e-12


def test_run_momentum_pnl_picks_the_trender_net_of_cost():
    # A trends up (+10%/mo), B is flat. Top-1 dual momentum must hold A
    # and earn A's next-month return, minus the entry turnover cost.
    n = 6
    a = [100 * 1.1 ** i for i in range(n)]
    b = [50.0] * n
    px = pd.DataFrame({"A": a, "B": b}, index=_month_index(n))
    cfg = MomentumConfig(lookback=2, top_k=1, skip=1, cost_bps=10.0, absolute_filter=True)
    res = run_momentum(px, cfg)

    # First decision is at row lookback+skip = 3, realized at row 4.
    first_ret = res.monthly_returns.iloc[0]
    a_next_ret = a[4] / a[3] - 1                       # 0.10
    entry_cost = res.turnover.iloc[0] * cfg.cost_bps / 1e4
    assert res.turnover.iloc[0] == 1.0                 # bought A from cash, weight 1.0
    assert abs(first_ret - (a_next_ret - entry_cost)) < 1e-12


def test_run_momentum_all_negative_goes_to_cash():
    # Every asset falling: dual momentum holds nothing, return is exactly
    # 0 (cash), never a negative "least-bad" pick.
    n = 6
    px = pd.DataFrame(
        {"A": [100 * 0.9 ** i for i in range(n)], "B": [80 * 0.95 ** i for i in range(n)]},
        index=_month_index(n),
    )
    cfg = MomentumConfig(lookback=2, top_k=1, skip=1, cost_bps=10.0, absolute_filter=True)
    res = run_momentum(px, cfg)
    assert (res.monthly_returns == 0.0).all()
    assert (res.weights.sum(axis=1) == 0.0).all()


def test_latest_target_weights_uses_only_last_row():
    n = 6
    px = pd.DataFrame(
        {"A": [100 * 1.1 ** i for i in range(n)], "B": [50.0] * n},
        index=_month_index(n),
    )
    cfg = MomentumConfig(lookback=2, top_k=1, skip=1, absolute_filter=True)
    w = latest_target_weights(px, cfg)
    assert w["A"] == 1.0 and w["B"] == 0.0


def test_summarize_known_monthly_returns():
    # +2% then -1% for 12 months alternating; check the plumbing.
    r = pd.Series([0.02, -0.01] * 6, index=_month_index(12))
    m = summarize(r, rf_annual=0.0)
    assert m["n_months"] == 12
    assert m["win_rate"] == 0.5
    assert m["max_drawdown"] <= 0.0
    # compounded 6*(1.02*0.99) growth is positive
    assert m["total_return"] > 0
    assert m["worst_month"] == -0.01 and m["best_month"] == 0.02


def test_summarize_empty():
    assert summarize(pd.Series([], dtype=float)) == {"n_months": 0}


def test_sharpe_positive_for_steady_gains():
    r = pd.Series([0.01] * 24, index=_month_index(24))
    m = summarize(r, rf_annual=0.0)
    # zero variance -> Sharpe defined as 0 by convention here, not inf
    assert m["sharpe"] == 0.0
    r2 = pd.Series([0.01, 0.02] * 12, index=_month_index(24))
    assert summarize(r2)["sharpe"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")
