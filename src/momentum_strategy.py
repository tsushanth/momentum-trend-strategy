"""Cross-sectional + absolute ("dual") momentum on a multi-asset ETF
basket. This is a deliberately standard, textbook formulation so the
result is interpretable against the known momentum literature -- nothing
exotic invented here.

The signal (Jegadeesh-Titman 1993 relative strength, cross-asset variant
per Asness-Moskowitz-Pedersen 2013; absolute-momentum overlay per
Antonacci 2014 "Dual Momentum"):

  1. Each month-end, for every asset compute its trailing L-month total
     return, ending `skip` months ago:
         mom_i = P_i[t - skip] / P_i[t - skip - L] - 1
     The 1-month skip is the standard guard against short-term reversal
     (last month's winners tend to bounce back down); "12-1" momentum is
     L=12, skip=1.
  2. CROSS-SECTIONAL: rank assets by mom, take the top K.
  3. ABSOLUTE (dual-momentum overlay, optional): of those K, only hold
     the ones whose own mom is still positive; the rest of the K equal-
     weight slots go to cash (0 return). This is what turns momentum
     into a trend follower with a downside brake -- in a broad selloff
     every asset's mom goes negative and the book moves to cash instead
     of "best of a bad bunch".
  4. Hold equal-weight (1/K per slot) until next month-end, then repeat.

Transaction costs are charged on turnover every rebalance and are NOT a
tunable parameter -- see COST note in run_momentum.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumConfig:
    lookback: int          # L: momentum formation window, in months
    top_k: int             # K: number of equal-weight slots
    skip: int = 1          # months skipped at the recent end (reversal guard)
    cost_bps: float = 10.0 # round-trip cost proxy, basis points per unit turnover
    absolute_filter: bool = True  # dual-momentum cash overlay


@dataclass
class BacktestResult:
    monthly_returns: pd.Series   # net portfolio return, indexed by the month it is realized
    weights: pd.DataFrame        # target weights decided at each rebalance (index = decision month)
    turnover: pd.Series          # L1 turnover charged at each rebalance
    config: MomentumConfig


def compute_momentum(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Trailing L-month total return ending `skip` months back, for each
    month-end row. Rows where the full L+skip history isn't available are
    NaN. `prices` must already be month-end (see momentum_data.to_month_end).
    """
    end = prices.shift(skip)
    start = prices.shift(skip + lookback)
    return end / start - 1.0


def target_weights(mom_row: pd.Series, top_k: int, absolute_filter: bool) -> pd.Series:
    """Weights for a single rebalance from one row of momentum scores.
    Equal-weight the top K; under the absolute filter, any of those K
    with non-positive momentum are dropped to cash (their weight simply
    stays 0, so the invested fraction can be < 1).
    """
    w = pd.Series(0.0, index=mom_row.index)
    ranked = mom_row.dropna().sort_values(ascending=False)
    if ranked.empty:
        return w
    picks = ranked.head(top_k)
    if absolute_filter:
        picks = picks[picks > 0]
    w.loc[picks.index] = 1.0 / top_k  # cash slots keep weight 0
    return w


def run_momentum(prices: pd.DataFrame, config: MomentumConfig,
                 start: int = 0, end: int | None = None) -> BacktestResult:
    """Walk month-end prices[start:end] and simulate the monthly-
    rebalanced dual-momentum book.

    Returns are realized one month after each decision: weights chosen
    at month t earn the t -> t+1 asset returns. Transaction cost is
    charged at the decision month against the L1 turnover from the prior
    weights.

    COST note: cost_bps is a fixed, conservative round-trip proxy
    (default 10bps of turned-over notional). It models the bid/ask +
    slippage a retail order actually pays on these liquid ETFs even
    though Alpaca charges no commission. It is intentionally not part of
    the parameter search -- you must never be able to tune your costs
    away to make a backtest look good.
    """
    if end is None:
        end = len(prices)
    px = prices.iloc[start:end]
    mom = compute_momentum(px, config.lookback, config.skip)
    asset_ret = px.pct_change()  # simple month-over-month return per asset

    first = config.lookback + config.skip  # first row with a valid signal
    prev_w = pd.Series(0.0, index=px.columns)
    cost_rate = config.cost_bps / 1e4

    ret_index, ret_vals, turn_vals, weight_rows, weight_idx = [], [], [], [], []
    # Decide at row t, realize over t -> t+1, so stop at len-1.
    for t in range(first, len(px) - 1):
        w = target_weights(mom.iloc[t], config.top_k, config.absolute_filter)
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * cost_rate

        gross = float((w * asset_ret.iloc[t + 1]).sum())  # cash slots contribute 0
        net = gross - cost

        ret_index.append(px.index[t + 1])
        ret_vals.append(net)
        turn_vals.append(turnover)
        weight_rows.append(w)
        weight_idx.append(px.index[t])
        prev_w = w

    return BacktestResult(
        monthly_returns=pd.Series(ret_vals, index=pd.DatetimeIndex(ret_index), name="net_return"),
        weights=pd.DataFrame(weight_rows, index=pd.DatetimeIndex(weight_idx)),
        turnover=pd.Series(turn_vals, index=pd.DatetimeIndex(weight_idx), name="turnover"),
        config=config,
    )


def latest_target_weights(prices: pd.DataFrame, config: MomentumConfig) -> pd.Series:
    """Weights the strategy would hold RIGHT NOW given month-end prices
    through today -- i.e. the signal from the most recent complete row.
    Used by the live executor so the traded book matches the backtested
    rule exactly. No look-ahead: uses only the last available month-end.
    """
    mom = compute_momentum(prices, config.lookback, config.skip)
    return target_weights(mom.iloc[-1], config.top_k, config.absolute_filter)
