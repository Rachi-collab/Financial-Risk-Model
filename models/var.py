import numpy as np
import pandas as pd
from scipy import stats


# ── 1. Historical Simulation ──────────────────────────────────────────────────

def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Historical VaR at *confidence* level over *horizon_days*.
    Sign convention: returned value is positive (a loss magnitude).
    """
    scaled = returns * np.sqrt(horizon_days)
    var = -np.percentile(scaled.dropna(), (1 - confidence) * 100)
    return float(var)


def cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Conditional VaR (Expected Shortfall) via historical method."""
    scaled = returns * np.sqrt(horizon_days)
    cutoff = np.percentile(scaled.dropna(), (1 - confidence) * 100)
    tail = scaled[scaled <= cutoff]
    cvar = -tail.mean()
    return float(cvar)


# ── 2. Parametric (Gaussian) ──────────────────────────────────────────────────

def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Parametric VaR assuming normally distributed returns.
    VaR = -(μ·h − z·σ·√h)  where h = horizon_days.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu * horizon_days + z * sigma * np.sqrt(horizon_days))
    return float(var)


def cvar_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Parametric CVaR (closed-form for normal distribution)."""
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    pdf_z = stats.norm.pdf(z)
    cvar = -(mu * horizon_days - sigma * np.sqrt(horizon_days) * pdf_z / (1 - confidence))
    return float(cvar)


# ── 3. Monte Carlo ────────────────────────────────────────────────────────────

def var_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Monte Carlo VaR & CVaR via Geometric Brownian Motion.

    Returns
    -------
    (var, cvar) as positive loss magnitudes
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std()

    # Simulate log-return paths
    shocks = rng.normal(
        loc=(mu - 0.5 * sigma ** 2) * horizon_days,
        scale=sigma * np.sqrt(horizon_days),
        size=n_simulations,
    )
    sim_returns = pd.Series(shocks)

    var = var_historical(sim_returns, confidence, horizon_days=1)
    cvar = cvar_historical(sim_returns, confidence, horizon_days=1)
    return var, cvar


# ── Summary helper ────────────────────────────────────────────────────────────

def var_summary(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Return a tidy DataFrame comparing all three VaR methods."""
    mc_var, mc_cvar = var_monte_carlo(returns, confidence, horizon_days)
    rows = [
        {
            "Method": "Historical",
            "VaR": var_historical(returns, confidence, horizon_days),
            "CVaR": cvar_historical(returns, confidence, horizon_days),
        },
        {
            "Method": "Parametric",
            "VaR": var_parametric(returns, confidence, horizon_days),
            "CVaR": cvar_parametric(returns, confidence, horizon_days),
        },
        {
            "Method": "Monte Carlo",
            "VaR": mc_var,
            "CVaR": mc_cvar,
        },
    ]
    df = pd.DataFrame(rows).set_index("Method")
    df["Confidence"] = f"{confidence*100:.0f}%"
    df["Horizon (days)"] = horizon_days
    return df.round(6)
