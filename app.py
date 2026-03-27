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

initial_fcf = st.sidebar.number_input("Initial FCF", value=1000.0)
growth_rate = st.sidebar.slider("DCF Growth Rate (%)", 0.0, 50.0, 20.0) / 100
wacc = st.sidebar.slider("WACC (%)", 5.0, 20.0, 12.0) / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 10.0, 4.0) / 100

# ML input
st.sidebar.subheader("ML Input (Historical FCF)")
fcf_input = st.sidebar.text_input("Enter past FCF values", "500,700,900,1200")

damping = st.sidebar.slider("ML Damping Factor (λ)", 0.6, 1.0, 0.85)

simulations = st.sidebar.slider("Monte Carlo Runs", 100, 3000, 500)
volatility = st.sidebar.slider("Simulation Volatility (%)", 1, 30, 10) / 100

run_button = st.sidebar.button("Run Simulation")

# -----------------------------
# CORE PROJECTION ENGINE (KEY)
# -----------------------------

def project_fcf(start_fcf, growth, terminal_growth, years, damping=0.85):
    fcf_list = []
    current_fcf = start_fcf

    for t in range(1, years + 1):
        # 🔥 Growth convergence (like Excel)
        adjusted_growth = terminal_growth + (growth - terminal_growth) * (damping ** t)

        current_fcf = current_fcf * (1 + adjusted_growth)
        fcf_list.append(current_fcf)

    return np.array(fcf_list)


def discounted_value(cashflows, wacc):
    return sum([
        cf / ((1 + wacc) ** (i + 1))
        for i, cf in enumerate(cashflows)
    ])


def compute_terminal_value(last_fcf, wacc, terminal_growth, years):
    tv = (last_fcf * (1 + terminal_growth)) / (wacc - terminal_growth + 0.01)
    return tv / ((1 + wacc) ** years)

# -----------------------------
# ML GROWTH ESTIMATION
# -----------------------------

try:
    fcf_history = list(map(float, fcf_input.split(",")))
except:
    st.error("Invalid FCF input")
    st.stop()

growth_rates = [
    (fcf_history[i] - fcf_history[i-1]) / fcf_history[i-1]
    for i in range(1, len(fcf_history))
]

base_growth = np.mean(growth_rates)
base_growth = min(base_growth, 0.25)

# Blend ML + DCF
ml_growth = 0.5 * base_growth + 0.5 * growth_rate

# -----------------------------
# VALUATIONS
# -----------------------------

# DCF (now corrected)
dcf_fcf = project_fcf(initial_fcf, growth_rate, terminal_growth, years)
dcf_val = discounted_value(dcf_fcf, wacc)
dcf_val += compute_terminal_value(dcf_fcf[-1], wacc, terminal_growth, years)

# ML (same engine, different growth)
ml_fcf = project_fcf(initial_fcf, ml_growth, terminal_growth, years)
ml_val = discounted_value(ml_fcf, wacc)
ml_val += compute_terminal_value(ml_fcf[-1], wacc, terminal_growth, years)

# -----------------------------
# OUTPUT
# -----------------------------

st.subheader("📌 Base Valuation")

col1, col2 = st.columns(2)
col1.metric("DCF Value", f"{dcf_val:,.2f}")
col2.metric("ML-Based Value", f"{ml_val:,.2f}")

st.write(f"📊 ML Growth Rate: {ml_growth:.2%}")
st.write(f"📉 DCF vs ML Difference: {(dcf_val - ml_val)/dcf_val:.2%}")

# -----------------------------
# FCF TABLE
# -----------------------------

st.subheader("📈 Projected Cash Flows")

df = pd.DataFrame({
    "Year": range(1, years + 1),
    "DCF FCF": dcf_fcf,
    "ML FCF": ml_fcf
})

st.dataframe(df)

# -----------------------------
# MONTE CARLO
# -----------------------------

@st.cache_data
def run_simulation(base_fcf, wacc, terminal_growth, years, simulations, volatility):
    results = []

    for _ in range(simulations):
        noise = np.random.normal(0, volatility, len(base_fcf))
        sim_fcf = base_fcf * (1 + noise)

        val = discounted_value(sim_fcf, wacc)
        val += compute_terminal_value(sim_fcf[-1], wacc, terminal_growth, years)

        results.append(val)

    return np.array(results)


if run_button:
    st.subheader("🎲 Monte Carlo Simulation")

    results = run_simulation(ml_fcf, wacc, terminal_growth, years, simulations, volatility)

    mean_val = np.mean(results)
    p5 = np.percentile(results, 5)
    p95 = np.percentile(results, 95)

    col3, col4, col5 = st.columns(3)
    col3.metric("Mean Value", f"{mean_val:,.2f}")
    col4.metric("5th Percentile", f"{p5:,.2f}")
    col5.metric("95th Percentile", f"{p95:,.2f}")

    fig, ax = plt.subplots()
    ax.hist(results, bins=30)
    ax.axvline(mean_val)
    ax.axvline(p5)
    ax.axvline(p95)

    st.pyplot(fig)

else:
    st.info("Click 'Run Simulation'")
