"""Real historical price data via yfinance -- same free-daily-bars
pattern as mm-backtester's pairs_data.py. Genuine market data, not a
synthetic generator with injected outcomes.

Universe choice (documented in the README): a diversified multi-asset
basket of liquid ETFs spanning US/international/EM equity, government
and corporate bonds, gold, broad commodities, and REITs. Cross-asset
momentum is the canonical application of the momentum literature
(Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere";
Antonacci 2014 "Dual Momentum") -- assets that trend for macro reasons,
with low enough correlation that a relative-strength rotation has
something real to rotate between. All tickers here have daily history
back to at least 2007, giving a long walk-forward window.
"""
import numpy as np
import pandas as pd
import yfinance as yf

# 10 liquid, low-cost ETFs across asset classes. Kept deliberately small
# and diversified: momentum needs a cross-section it can rank, but a
# huge single-asset-class universe (e.g. 50 tech stocks) would just be a
# leveraged bet on that class, not a diversifying trend follower.
DEFAULT_UNIVERSE = [
    "SPY",   # US large-cap equity
    "EFA",   # developed ex-US equity
    "EEM",   # emerging-market equity
    "IWM",   # US small-cap equity
    "TLT",   # 20+yr US Treasuries
    "IEF",   # 7-10yr US Treasuries
    "LQD",   # investment-grade corporate bonds
    "GLD",   # gold
    "DBC",   # broad commodities
    "VNQ",   # US REITs
]


def load_universe(symbols=None, start="2007-01-01", end=None) -> pd.DataFrame:
    """Return a DataFrame of adjusted daily closes indexed by date, one
    column per symbol. auto_adjust=True folds dividends/splits into the
    price so total-return momentum (not just price momentum) is what we
    actually rank -- for bond and REIT ETFs the dividend is a big chunk
    of total return, so ignoring it would distort the signal.

    Rows with any missing symbol are dropped so every asset is aligned
    on the same trading calendar; the effective start date is therefore
    the latest inception date across the basket.
    """
    if symbols is None:
        symbols = DEFAULT_UNIVERSE
    raw = yf.download(symbols, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close[symbols]  # preserve caller's column order
    close = close.dropna(how="any")
    return close


def to_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily closes to month-end (last observation each month).
    Momentum here is a monthly-rebalanced strategy, so the whole
    backtest runs on month-end prices -- daily bars are only the raw
    input we downsample from.
    """
    return daily.resample("ME").last().dropna(how="any")
