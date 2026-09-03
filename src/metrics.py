"""Performance metrics for a monthly return series. Unlike
mm-backtester's metrics.py (which summarizes discrete trades), momentum
is a continuously-held portfolio, so the natural unit is the periodic
(monthly) return stream.

Sharpe here is excess-return over a risk-free rate, annualized by
sqrt(12) as is standard for monthly data. Everything is net of costs --
the return series handed in has already had transaction costs subtracted
per rebalance (see momentum_strategy.run_momentum).
"""
import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def summarize(returns: pd.Series, rf_annual: float = 0.0) -> dict:
    """returns: monthly net return series. rf_annual: annual risk-free
    rate (e.g. 0.02 for 2%) used only for the Sharpe excess-return term.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n == 0:
        return {"n_months": 0}

    rf_monthly = (1 + rf_annual) ** (1 / MONTHS_PER_YEAR) - 1
    excess = r - rf_monthly

    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = n / MONTHS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    vol_annual = float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if n > 1 else 0.0
    if n > 1 and r.std(ddof=1) > 0:
        sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    return {
        "n_months": n,
        "total_return": total_return,
        "cagr": cagr,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": float((r > 0).mean()),
        "best_month": float(r.max()),
        "worst_month": float(r.min()),
    }
