import numpy as np
import pandas as pd

# ── EWMA ──────────────────────────────────────────────────────────────────────


def ewma_volatility(
    returns: pd.Series,
    lam: float = 0.94,
    annualise: bool = True,
) -> pd.Series:
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics standard).
    λ = 0.94 for daily data (JP Morgan convention).
    """
    sq_ret = returns**2
    ewma_var = sq_ret.ewm(com=(1 - lam) / lam, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    if annualise:
        ewma_vol *= np.sqrt(252)
    ewma_vol.name = f"EWMA_vol(λ={lam})"
    return ewma_vol


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────


def garch_volatility(
    returns: pd.Series,
    annualise: bool = True,
) -> pd.Series | None:
    """
    Fit a GARCH(1,1) model and return the conditional volatility series.
    Requires the `arch` library — returns None if not installed.
    """
    try:
        from arch import arch_model  # type: ignore
    except ImportError:
        print("[volatility] `arch` package not found. Install with: pip install arch")
        return None

    model = arch_model(returns * 100, vol="Garch", p=1, q=1, rescale=False)
    res = model.fit(disp="off")
    cond_vol = res.conditional_volatility / 100  # back to decimal
    if annualise:
        cond_vol *= np.sqrt(252)
    cond_vol.name = "GARCH(1,1)_vol"
    return cond_vol


# ── Rolling ───────────────────────────────────────────────────────────────────


def rolling_vol_bands(
    returns: pd.Series,
    windows: list[int] = [21, 63, 252],
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame of rolling volatilities for multiple windows.
    windows: [21, 63, 252] ≈ [1-month, 1-quarter, 1-year]
    """
    frames = {}
    for w in windows:
        vol = returns.rolling(w).std()
        if annualise:
            vol *= np.sqrt(252)
        frames[f"{w}d_vol"] = vol
    return pd.DataFrame(frames)
