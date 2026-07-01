import pandas as pd
from data import load_sample_portfolio, log_returns, portfolio_returns
from models import var_summary, ewma_volatility
from analysis import performance_summary, max_drawdown, kupiec_pof_test
from analysis.backtesting import rolling_var


def main():
    print("=" * 60)
    print("  FINANCIAL RISK MODEL  —  Full Pipeline Report")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────
    print("\n[1/5] Loading price data …")
    prices, weights = load_sample_portfolio()
    returns_df = log_returns(prices)
    port_ret = portfolio_returns(returns_df, weights)
    print(f"      Loaded {len(prices)} trading days for {list(prices.columns)}")

    # ── 2. Performance ────────────────────────────────────────────
    print("\n[2/5] Performance summary")
    summary = performance_summary(port_ret, risk_free_rate=0.05)
    print(summary.to_string())

    # ── 3. VaR / CVaR ─────────────────────────────────────────────
    print("\n[3/5] VaR & CVaR (95%, 1-day)")
    var_df = var_summary(port_ret, confidence=0.95, horizon_days=1)
    print(var_df.to_string())

    print("\n      VaR & CVaR (95%, 5-day)")
    var_df5 = var_summary(port_ret, confidence=0.95, horizon_days=5)
    print(var_df5.to_string())

    # ── 4. EWMA Volatility ────────────────────────────────────────
    print("\n[4/5] Latest EWMA volatility (annualised)")
    ewma = ewma_volatility(port_ret)
    print(f"      Current EWMA vol: {ewma.iloc[-1]:.2%}")

    # ── 5. Backtest ───────────────────────────────────────────────
    print("\n[5/5] Kupiec backtest (historical VaR, 252-day window, 95%)")
    var_roll = rolling_var(port_ret, confidence=0.95, method="historical", window=252)
    kupiec = kupiec_pof_test(port_ret, var_roll, confidence=0.95)
    for k, v in kupiec.items():
        print(f"      {k:25s}: {v}")

    print("\n" + "=" * 60)
    print("  Done. To launch the interactive dashboard run:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
