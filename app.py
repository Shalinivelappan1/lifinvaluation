import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fintech Valuation Simulator", layout="wide")

st.title("📊 Fintech Valuation Simulator (DCF + ML + Monte Carlo)")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Model Inputs")

years = st.sidebar.slider("Projection Years", 3, 15, 5)

# DCF Inputs
initial_fcf = st.sidebar.number_input("Initial FCF", value=1000.0)
growth_rate = st.sidebar.slider("DCF Growth Rate (%)", 0.0, 50.0, 20.0) / 100
wacc = st.sidebar.slider("WACC (%)", 5.0, 20.0, 12.0) / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 10.0, 4.0) / 100

# ML Inputs
st.sidebar.subheader("ML Input (Historical FCF)")
fcf_input = st.sidebar.text_input(
    "Enter past FCF values (comma separated)",
    "500,700,900,1200"
)

damping = st.sidebar.slider("ML Damping Factor (λ)", 0.6, 1.0, 0.80)

# Monte Carlo
simulations = st.sidebar.slider("Monte Carlo Runs", 100, 3000, 500)
volatility = st.sidebar.slider("Simulation Volatility (%)", 1, 30, 10) / 100

run_button = st.sidebar.button("Run Simulation")

# -----------------------------
# FUNCTIONS
# -----------------------------

def dcf_valuation(fcf, growth, wacc, terminal_growth, years):
    cashflows = []
    for t in range(1, years + 1):
        fcf = fcf * (1 + growth)
        cashflows.append(fcf / ((1 + wacc) ** t))

    # Slight stabilization for long horizons
    terminal_value = (fcf * (1 + terminal_growth)) / (wacc - terminal_growth + 0.01)
    terminal_discounted = terminal_value / ((1 + wacc) ** years)

    return sum(cashflows) + terminal_discounted


# ✅ FIXED ML FUNCTION (WITH CONVERGENCE)
def ml_fcf_prediction(fcf_history, years, damping, dcf_growth, terminal_growth):
    growth_rates = []

    for i in range(1, len(fcf_history)):
        growth_rates.append(
            (fcf_history[i] - fcf_history[i-1]) / fcf_history[i-1]
        )

    base_growth = np.mean(growth_rates)

    # Cap excessive growth
    base_growth = min(base_growth, 0.25)

    # Blend ML + DCF growth
    blended_growth = 0.5 * base_growth + 0.5 * dcf_growth

    predictions = []
    last_fcf = fcf_history[-1]

    for t in range(1, years + 1):
        # 🔥 KEY FIX: Converge toward terminal growth
        adjusted_growth = terminal_growth + (blended_growth - terminal_growth) * (damping ** t)

        last_fcf = last_fcf * (1 + adjusted_growth)
        predictions.append(last_fcf)

    return np.array(predictions), blended_growth


def discounted_value(cashflows, wacc):
    return sum([
        cf / ((1 + wacc) ** (i + 1))
        for i, cf in enumerate(cashflows)
    ])


@st.cache_data
def run_simulation(ml_predicted_fcf, wacc, terminal_growth, years, simulations, volatility):
    results = []

    for _ in range(simulations):
        noise = np.random.normal(0, volatility, len(ml_predicted_fcf))
        simulated_fcf = ml_predicted_fcf * (1 + noise)

        terminal_value = (simulated_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth + 0.01)
        terminal_discounted = terminal_value / ((1 + wacc) ** years)

        value = discounted_value(simulated_fcf, wacc) + terminal_discounted
        results.append(value)

    return np.array(results)

# -----------------------------
# DATA PROCESSING
# -----------------------------

try:
    fcf_history = list(map(float, fcf_input.split(",")))
except:
    st.error("Invalid FCF input format")
    st.stop()

# -----------------------------
# BASE VALUATION
# -----------------------------

st.subheader("📌 Base Valuation")

dcf_val = dcf_valuation(initial_fcf, growth_rate, wacc, terminal_growth, years)

ml_predicted_fcf, blended_growth = ml_fcf_prediction(
    fcf_history, years, damping, growth_rate, terminal_growth
)

# ML valuation WITH terminal value
ml_terminal_value = (ml_predicted_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth + 0.01)
ml_terminal_discounted = ml_terminal_value / ((1 + wacc) ** years)

ml_val = discounted_value(ml_predicted_fcf, wacc) + ml_terminal_discounted

col1, col2 = st.columns(2)
col1.metric("DCF Value", f"{dcf_val:,.2f}")
col2.metric("ML-Based Value", f"{ml_val:,.2f}")

# Insights
st.write(f"📊 Blended ML Growth Rate: {blended_growth:.2%}")
st.write(f"📉 DCF vs ML Difference: {(dcf_val - ml_val)/dcf_val:.2%}")

# -----------------------------
# ML FORECAST
# -----------------------------

st.subheader("📈 ML Forecasted Cash Flows")

df_ml = pd.DataFrame({
    "Year": range(1, years + 1),
    "Predicted FCF": ml_predicted_fcf
})

st.dataframe(df_ml)

# -----------------------------
# MONTE CARLO
# -----------------------------

if run_button:
    st.subheader("🎲 Monte Carlo Simulation")

    results = run_simulation(
        ml_predicted_fcf, wacc, terminal_growth, years, simulations, volatility
    )

    mean_val = np.mean(results)
    p5 = np.percentile(results, 5)
    p95 = np.percentile(results, 95)

    col3, col4, col5 = st.columns(3)
    col3.metric("Mean Value", f"{mean_val:,.2f}")
    col4.metric("5th Percentile", f"{p5:,.2f}")
    col5.metric("95th Percentile", f"{p95:,.2f}")

    # Histogram
    st.subheader("📊 Valuation Distribution")

    fig, ax = plt.subplots()
    ax.hist(results, bins=30)

    ax.axvline(mean_val, linestyle="dashed")
    ax.axvline(p5, linestyle="dotted")
    ax.axvline(p95, linestyle="dotted")

    ax.set_xlabel("Valuation")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # Interpretation
    st.subheader("🧠 Interpretation")

    st.write(f"""
    - Mean valuation: {mean_val:,.2f}
    - Downside risk (5th percentile): {p5:,.2f}
    - Upside potential (95th percentile): {p95:,.2f}

    ML-based valuation converges toward DCF as growth stabilizes over time,
    reflecting realistic long-term economic behavior.
    """)

else:
    st.info("Click 'Run Simulation' to generate Monte Carlo results.")
