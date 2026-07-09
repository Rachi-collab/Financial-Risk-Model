import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data import (
    fetch_prices,
    log_returns,
    simple_returns,
    portfolio_returns,
    rolling_volatility,
)
from models import var_summary, ewma_volatility, rolling_vol_bands
from analysis import (
    performance_summary,
    drawdown_series,
    max_drawdown,
    correlation_matrix,
    rolling_var,
    count_violations,
    kupiec_pof_test,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card {
        background: #0e1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
    }
    .stMetric label { font-size: 0.78rem; color: #8b949e; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

PRESETS = {
    "Tech Heavy": {
        "AAPL": 0.30,
        "MSFT": 0.25,
        "GOOGL": 0.20,
        "NVDA": 0.15,
        "META": 0.10,
    },
    "Diversified": {"AAPL": 0.20, "MSFT": 0.20, "JPM": 0.20, "XOM": 0.20, "JNJ": 0.20},
    "Finance & Energy": {"JPM": 0.25, "BAC": 0.25, "XOM": 0.25, "CVX": 0.25},
}

preset = st.sidebar.selectbox("Portfolio Preset", list(PRESETS.keys()))
default_tickers = ", ".join(PRESETS[preset].keys())
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2020-01-01"))
confidence = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
horizon = st.sidebar.selectbox("VaR Horizon (days)", [1, 5, 10], index=0)
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100

# Weights
st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Weights")
preset_weights = PRESETS[preset]
raw_weights = {}
for t in tickers:
    default_w = preset_weights.get(t, 1 / len(tickers))
    raw_weights[t] = st.sidebar.slider(t, 0.0, 1.0, float(round(default_w, 2)), 0.01)

total_w = sum(raw_weights.values())
weights = (
    {t: w / total_w for t, w in raw_weights.items()}
    if total_w > 0
    else {t: 1 / len(tickers) for t in tickers}
)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(tickers_key, start):
    tickers_list = tickers_key.split("|")
    prices = fetch_prices(tickers_list, start=str(start))
    return prices


prices = load_data("|".join(sorted(tickers)), start_date)
returns_df = log_returns(prices)
port_ret = portfolio_returns(returns_df, weights)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Market Risk Dashboard")
st.caption(f"Portfolio: {' · '.join(f'{t} ({w:.0%})' for t, w in weights.items())}")

# ── Top metrics ───────────────────────────────────────────────────────────────
summary = performance_summary(port_ret, rf_rate)
col1, col2, col3, col4, col5 = st.columns(5)

ann_ret = port_ret.mean() * 252
ann_vol = port_ret.std() * np.sqrt(252)
_, max_dd = max_drawdown(port_ret)
sharpe = float(summary["Sharpe Ratio"])
var_hist = float(ann_vol / np.sqrt(252) * 1.645)  # quick daily VaR

col1.metric("Annualised Return", f"{ann_ret:.2%}", delta=None)
col2.metric("Annualised Vol", f"{ann_vol:.2%}")
col3.metric("Sharpe Ratio", summary["Sharpe Ratio"])
col4.metric("Max Drawdown", summary["Max Drawdown"])
col5.metric(f"1-Day VaR ({confidence:.0%})", f"{var_hist:.2%}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Returns & Prices",
        "🎯 VaR Analysis",
        "📉 Drawdown",
        "🔗 Correlations",
        "🧪 Backtest",
    ]
)

# ─ Tab 1: Returns ─────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Cumulative Returns")
        cum = (1 + returns_df).cumprod()
        port_cum = (1 + port_ret).cumprod()
        fig = go.Figure()
        for col in cum.columns:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col, opacity=0.6))
        fig.add_trace(
            go.Scatter(
                x=port_cum.index,
                y=port_cum,
                name="Portfolio",
                line=dict(width=3, color="#f0c040"),
            )
        )
        fig.update_layout(template="plotly_dark", height=360, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Rolling Volatility (EWMA)")
        ewma_vol = ewma_volatility(port_ret)
        roll_vols = rolling_vol_bands(port_ret)
        fig2 = go.Figure()
        for col in roll_vols.columns:
            fig2.add_trace(
                go.Scatter(x=roll_vols.index, y=roll_vols[col], name=col, opacity=0.5)
            )
        fig2.add_trace(
            go.Scatter(
                x=ewma_vol.index,
                y=ewma_vol,
                name="EWMA λ=0.94",
                line=dict(width=2, color="#f0c040"),
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(t=20),
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Return Distribution")
    fig3 = px.histogram(
        port_ret, nbins=80, template="plotly_dark", color_discrete_sequence=["#4f86f7"]
    )
    fig3.update_layout(height=280, margin=dict(t=20), xaxis_tickformat=".2%")
    st.plotly_chart(fig3, use_container_width=True)


# ─ Tab 2: VaR ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(
        f"VaR & CVaR Comparison — {confidence:.0%} confidence, {horizon}-day horizon"
    )
    var_df = var_summary(port_ret, confidence, horizon)
    st.dataframe(
        var_df.style.format({"VaR": "{:.4%}", "CVaR": "{:.4%}"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Return Distribution with VaR cutoffs")
    from models.var import var_historical, var_parametric

    var_h = var_historical(port_ret, confidence, horizon)
    var_p = var_parametric(port_ret, confidence, horizon)

    fig4 = go.Figure()
    fig4.add_trace(
        go.Histogram(
            x=port_ret, nbinsx=80, name="Returns", marker_color="#4f86f7", opacity=0.7
        )
    )
    for val, name, color in [
        (-var_h, f"Hist VaR", "#ff4b4b"),
        (-var_p, f"Param VaR", "#ffa600"),
    ]:
        fig4.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="top right",
        )
    fig4.update_layout(template="plotly_dark", height=350, margin=dict(t=20))
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Performance Summary")
    st.dataframe(
        pd.DataFrame(summary).rename(columns={0: "Value"}), use_container_width=True
    )


# ─ Tab 3: Drawdown ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Drawdown Analysis")
    dd = drawdown_series(port_ret)
    wealth = (1 + port_ret).cumprod()
    trough_date, max_dd_val = max_drawdown(port_ret)

    fig5 = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06
    )
    fig5.add_trace(
        go.Scatter(
            x=wealth.index, y=wealth, name="Portfolio Value", line=dict(color="#4f86f7")
        ),
        row=1,
        col=1,
    )
    fig5.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#ff4b4b"),
            fillcolor="rgba(255,75,75,0.2)",
        ),
        row=2,
        col=1,
    )
    fig5.update_layout(template="plotly_dark", height=500, margin=dict(t=20))
    fig5.update_yaxes(tickformat=".1%", row=2)
    st.plotly_chart(fig5, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Max Drawdown", f"{max_dd_val:.2%}")
    col2.metric("Trough Date", str(trough_date.date()))
    col3.metric("Calmar Ratio", summary["Calmar Ratio"])


# ─ Tab 4: Correlations ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Asset Correlation Matrix")
    corr = correlation_matrix(returns_df)
    fig6 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template="plotly_dark",
    )
    fig6.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Scatter Matrix")
    fig7 = px.scatter_matrix(
        returns_df.dropna(),
        template="plotly_dark",
        dimensions=returns_df.columns.tolist(),
    )
    fig7.update_traces(marker=dict(size=2, opacity=0.4))
    fig7.update_layout(height=500, margin=dict(t=20))
    st.plotly_chart(fig7, use_container_width=True)


# ─ Tab 5: Backtest ────────────────────────────────────────────────────────────
with tab5:
    st.subheader("VaR Backtest — Rolling Window")
    roll_window = st.slider("Look-back window (days)", 126, 504, 252, 21)
    method = st.radio("VaR method", ["historical", "parametric"], horizontal=True)

    var_roll = rolling_var(port_ret, confidence, method, roll_window, horizon)
    violations_df = count_violations(port_ret, var_roll)

    fig8 = go.Figure()
    fig8.add_trace(
        go.Scatter(
            x=violations_df.index,
            y=violations_df["return"],
            name="Daily Return",
            line=dict(color="#4f86f7", width=1),
        )
    )
    fig8.add_trace(
        go.Scatter(
            x=violations_df.index,
            y=-violations_df["VaR"],
            name=f"{confidence:.0%} VaR",
            line=dict(color="#ffa600", dash="dash"),
        )
    )

    viol = violations_df[violations_df["violation"]]
    fig8.add_trace(
        go.Scatter(
            x=viol.index,
            y=viol["return"],
            mode="markers",
            name="Violation",
            marker=dict(color="red", size=6, symbol="x"),
        )
    )
    fig8.update_layout(
        template="plotly_dark", height=380, margin=dict(t=20), yaxis_tickformat=".2%"
    )
    st.plotly_chart(fig8, use_container_width=True)

    kupiec = kupiec_pof_test(port_ret, var_roll, confidence)
    k_df = pd.DataFrame.from_dict(kupiec, orient="index", columns=["Value"])
    st.subheader("Kupiec POF Test Results")
    st.dataframe(k_df, use_container_width=True)

    result_color = "green" if "PASS" in str(kupiec["result"]) else "red"
    st.markdown(f"**Result:** :{result_color}[{kupiec['result']}]")
