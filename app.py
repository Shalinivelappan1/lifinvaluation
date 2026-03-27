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

years = st.sidebar.slider("Projection Years", 3, 10, 5)

# DCF Inputs
initial_fcf = st.sidebar.number_input("Initial FCF", value=1000.0)
growth_rate = st.sidebar.slider("Growth Rate (%)", 0.0, 50.0, 20.0) / 100
wacc = st.sidebar.slider("WACC (%)", 5.0, 20.0, 12.0) / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 10.0, 4.0) / 100

# ML-style input
st.sidebar.subheader("ML Input (Historical FCF)")
fcf_input = st.sidebar.text_input(
    "Enter past FCF values (comma separated)",
    "500,700,900,1200"
)

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

    terminal_value = (fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    terminal_discounted = terminal_value / ((1 + wacc) ** years)

    return sum(cashflows) + terminal_discounted


# ML-style projection (trend-based, lightweight)
def ml_fcf_prediction(fcf_history, years):
    growth_rates = []

    for i in range(1, len(fcf_history)):
        growth_rates.append(
            (fcf_history[i] - fcf_history[i-1]) / fcf_history[i-1]
        )

    avg_growth = np.mean(growth_rates)

    predictions = []
    last_fcf = fcf_history[-1]

    for _ in range(years):
        last_fcf = last_fcf * (1 + avg_growth)
        predictions.append(last_fcf)

    return np.array(predictions)


def discounted_value(cashflows, wacc):
    return sum([
        cf / ((1 + wacc) ** (i + 1))
        for i, cf in enumerate(cashflows)
    ])


@st.cache_data
def run_simulation(ml_predicted_fcf, wacc, simulations, volatility):
    results = []

    for _ in range(simulations):
        noise = np.random.normal(0, volatility, len(ml_predicted_fcf))
        simulated_fcf = ml_predicted_fcf * (1 + noise)

        value = discounted_value(simulated_fcf, wacc)
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

ml_predicted_fcf = ml_fcf_prediction(fcf_history, years)
ml_val = discounted_value(ml_predicted_fcf, wacc)

col1, col2 = st.columns(2)
col1.metric("DCF Value", f"{dcf_val:,.2f}")
col2.metric("ML-Based Value", f"{ml_val:,.2f}")

# -----------------------------
# ML FORECAST TABLE
# -----------------------------

st.subheader("📈 ML Forecasted Cash Flows")

df_ml = pd.DataFrame({
    "Year": range(1, years + 1),
    "Predicted FCF": ml_predicted_fcf
})

st.dataframe(df_ml)

# -----------------------------
# MONTE CARLO (RUN ON BUTTON)
# -----------------------------

if run_button:
    st.subheader("🎲 Monte Carlo Simulation")

    results = run_simulation(ml_predicted_fcf, wacc, simulations, volatility)

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
    ax.hist(results, bins=40)
    ax.axvline(mean_val)
    ax.axvline(p5)
    ax.axvline(p95)

    ax.set_xlabel("Valuation")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # Interpretation
    st.subheader("🧠 Interpretation")

    st.write(f"""
    - Mean valuation: {mean_val:,.2f}
    - Downside risk (5th percentile): {p5:,.2f}
    - Upside potential (95th percentile): {p95:,.2f}

    This reflects valuation uncertainty under stochastic cash flow scenarios.
    """)

else:
    st.info("Click 'Run Simulation' to generate Monte Carlo results.")
