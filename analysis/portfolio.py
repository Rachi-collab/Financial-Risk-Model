import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ── Return / Risk ratios ──────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    annualise: bool = True,
) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free_rate / TRADING_DAYS
    if annualise:
        return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))
    return float(excess.mean() / excess.std())


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """Sortino ratio — penalises only downside deviation."""
    excess = returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0].std() * np.sqrt(TRADING_DAYS)
    annualised_excess = excess.mean() * TRADING_DAYS
    return float(annualised_excess / downside) if downside > 0 else np.nan


def calmar_ratio(
    returns: pd.Series,
) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    ann_return = returns.mean() * TRADING_DAYS
    _, max_dd = max_drawdown(returns)
    return float(ann_return / abs(max_dd)) if max_dd != 0 else np.nan


# ── Drawdown ──────────────────────────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown at each point in time."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    dd.name = "Drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> tuple[pd.Timestamp, float]:
    """Return (date_of_trough, max_drawdown_value)."""
    dd = drawdown_series(returns)
    trough_date = dd.idxmin()
    return trough_date, float(dd.min())


# ── Correlation & Covariance ──────────────────────────────────────────────────

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix."""
    return returns.corr()


def covariance_matrix(
    returns: pd.DataFrame, annualise: bool = True
) -> pd.DataFrame:
    cov = returns.cov()
    if annualise:
        cov *= TRADING_DAYS
    return cov


# ── Full summary ──────────────────────────────────────────────────────────────

def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> pd.Series:
    """One-stop performance fact sheet."""
    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    _, max_dd = max_drawdown(returns)

    metrics = {
        "Annualised Return": f"{ann_ret:.2%}",
        "Annualised Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns, risk_free_rate):.3f}",
        "Sortino Ratio": f"{sortino_ratio(returns, risk_free_rate):.3f}",
        "Calmar Ratio": f"{calmar_ratio(returns):.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Skewness": f"{returns.skew():.3f}",
        "Kurtosis (excess)": f"{returns.kurt():.3f}",
        "Total Observations": len(returns),
    }
    return pd.Series(metrics, name="Value")
