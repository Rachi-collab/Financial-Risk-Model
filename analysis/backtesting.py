import numpy as np
import pandas as pd
from scipy import stats

from models.var import var_historical, var_parametric


def rolling_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
    window: int = 252,
    horizon_days: int = 1,
) -> pd.Series:
    """
    Compute a rolling 1-day VaR estimate.

    Parameters
    ----------
    window   : look-back window in trading days (default 252 = 1 year)
    method   : 'historical' | 'parametric'
    """
    var_fn = var_historical if method == "historical" else var_parametric

    var_series = returns.rolling(window).apply(
        lambda r: var_fn(pd.Series(r), confidence, horizon_days),
        raw=False,
    )
    var_series.name = f"VaR_{method}_{confidence:.0%}"
    return var_series


def count_violations(
    returns: pd.Series,
    var_series: pd.Series,
) -> pd.DataFrame:
    """
    Return a DataFrame with a boolean 'violation' column.
    A violation occurs when the realised loss exceeds the predicted VaR.
    Note: returns are signed (negative = loss), VaR is positive (loss magnitude).
    """
    df = pd.DataFrame({"return": returns, "VaR": var_series}).dropna()
    df["violation"] = df["return"] < -df["VaR"]
    return df


def kupiec_pof_test(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures test.

    H₀: The true exception rate equals (1 - confidence).
    Rejects H₀ when the model is significantly mis-calibrated.

    Returns a dict with: n_obs, n_violations, expected_violations,
    violation_rate, LR_statistic, p_value, result.
    """
    df = count_violations(returns, var_series)
    n = len(df)
    v = df["violation"].sum()
    p = 1 - confidence  # expected violation probability
    p_hat = v / n  # observed

    # Log-likelihood ratio statistic
    if p_hat == 0 or p_hat == 1:
        lr = np.nan
        p_val = np.nan
    else:
        lr = -2 * (
            np.log(p**v * (1 - p) ** (n - v))
            - np.log(p_hat**v * (1 - p_hat) ** (n - v))
        )
        p_val = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "n_obs": n,
        "n_violations": int(v),
        "expected_violations": round(n * p, 1),
        "violation_rate": round(p_hat, 4),
        "LR_statistic": round(lr, 4) if not np.isnan(lr) else "N/A",
        "p_value": round(p_val, 4) if not np.isnan(p_val) else "N/A",
        "result": (
            "PASS — model is well-calibrated (fail to reject H₀)"
            if (not np.isnan(p_val) and p_val > 0.05)
            else "FAIL — model may be mis-calibrated (reject H₀ at 5%)"
        ),
    }
