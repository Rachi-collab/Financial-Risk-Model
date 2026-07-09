import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Continuously compounded (log) returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) returns: (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change().dropna()


def portfolio_returns(
    returns: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    """
    Compute weighted portfolio return series.

    Parameters
    ----------
    returns : pd.DataFrame  — individual asset returns (log or simple)
    weights : dict          — {ticker: weight}; must sum to ~1.0
    """
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    w /= w.sum()  # normalise just in case
    port_ret = returns.dot(w)
    port_ret.name = "Portfolio"
    return port_ret


def rolling_volatility(
    returns: pd.Series | pd.DataFrame, window: int = 21, annualise: bool = True
) -> pd.Series | pd.DataFrame:
    """Rolling standard deviation. Default window = 21 trading days (~1 month)."""
    vol = returns.rolling(window).std()
    if annualise:
        vol *= np.sqrt(252)
    return vol
