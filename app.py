import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fintech Valuation Lab", layout="wide")

st.title("📊 Fintech Valuation Lab (DCF + ML + Scenarios + Monte Carlo)")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Model Inputs")

years = st.sidebar.slider("Projection Years", 3, 15, 5)

initial_fcf = st.sidebar.number_input("Initial FCF", value=950.0)

growth_rate = st.sidebar.slider("Base Growth (%)", 0.0, 50.0, 18.0) / 100
wacc = st.sidebar.slider("WACC (%)", 5.0, 20.0, 11.0) / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 10.0, 4.5) / 100

# Capital structure
st.sidebar.subheader("Capital Structure")
debt = st.sidebar.number_input("Debt", value=2000.0)
cash = st.sidebar.number_input("Cash", value=500.0)
shares = st.sidebar.number_input("Shares Outstanding (millions)", value=100.0)

# ML input
st.sidebar.subheader("Historical Cash Flow Input")
fcf_input = st.sidebar.text_input("Enter past values", "450,650,900,1200")

damping = st.sidebar.slider("Growth Damping (λ)", 0.6, 1.0, 0.85)

# Monte Carlo
simulations = st.sidebar.slider("Simulation Runs", 100, 3000, 500)
volatility = st.sidebar.slider("Volatility (%)", 1, 30, 12) / 100

run_button = st.sidebar.button("Run Simulation")

# -----------------------------
# FUNCTIONS
# -----------------------------

def project_fcf(start_fcf, growth, terminal_growth, years, damping):
    fcf_list = []
    current_fcf = start_fcf

    for t in range(1, years + 1):
        adjusted_growth = terminal_growth + (growth - terminal_growth) * (damping ** t)
        current_fcf = current_fcf * (1 + adjusted_growth)
        fcf_list.append(current_fcf)

    return np.array(fcf_list)


def discounted_value(cashflows, wacc):
    return sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(cashflows))


def terminal_value(last_fcf, wacc, g, years):
    tv = (last_fcf * (1 + g)) / (wacc - g + 0.01)
    return tv / ((1 + wacc) ** years)


def equity_value(firm_value, debt, cash):
    return firm_value - debt + cash


# -----------------------------
# ML GROWTH
# -----------------------------
try:
    fcf_history = list(map(float, fcf_input.split(",")))
except:
    st.error("Invalid input")
    st.stop()

growth_rates = [(fcf_history[i] - fcf_history[i-1]) / fcf_history[i-1]
                for i in range(1, len(fcf_history))]

base_growth = min(np.mean(growth_rates), 0.25)
ml_growth = 0.5 * base_growth + 0.5 * growth_rate

# -----------------------------
# SCENARIOS
# -----------------------------

scenarios = {
    "Bear": growth_rate - 0.05,
    "Base": growth_rate,
    "Bull": growth_rate + 0.05
}

results_table = []

st.subheader("📊 Scenario Valuation")

for name, g in scenarios.items():

    dcf_fcf = project_fcf(initial_fcf, g, terminal_growth, years, damping)
    dcf_val = discounted_value(dcf_fcf, wacc)
    dcf_val += terminal_value(dcf_fcf[-1], wacc, terminal_growth, years)

    eq_val = equity_value(dcf_val, debt, cash)
    price = eq_val / shares

    results_table.append([name, dcf_val, eq_val, price])

df_scenarios = pd.DataFrame(results_table, columns=[
    "Scenario", "Firm Value", "Equity Value", "Share Price"
])

st.dataframe(df_scenarios)

# -----------------------------
# BASE MODEL OUTPUT
# -----------------------------

st.subheader("📌 Base Model (DCF vs ML)")

# DCF
dcf_fcf = project_fcf(initial_fcf, growth_rate, terminal_growth, years, damping)
dcf_val = discounted_value(dcf_fcf, wacc)
dcf_val += terminal_value(dcf_fcf[-1], wacc, terminal_growth, years)

# ML
ml_fcf = project_fcf(initial_fcf, ml_growth, terminal_growth, years, damping)
ml_val = discounted_value(ml_fcf, wacc)
ml_val += terminal_value(ml_fcf[-1], wacc, terminal_growth, years)

# Equity + Price
dcf_eq = equity_value(dcf_val, debt, cash)
ml_eq = equity_value(ml_val, debt, cash)

dcf_price = dcf_eq / shares
ml_price = ml_eq / shares

col1, col2 = st.columns(2)
col1.metric("DCF Share Price", f"{dcf_price:,.2f}")
col2.metric("ML Share Price", f"{ml_price:,.2f}")

st.write(f"📊 ML Growth Rate: {ml_growth:.2%}")

# -----------------------------
# MONTE CARLO
# -----------------------------

@st.cache_data
def run_simulation(base_fcf):
    vals = []

    for _ in range(simulations):
        noise = np.random.normal(0, volatility, len(base_fcf))
        sim_fcf = base_fcf * (1 + noise)

        val = discounted_value(sim_fcf, wacc)
        val += terminal_value(sim_fcf[-1], wacc, terminal_growth, years)

        eq = equity_value(val, debt, cash)
        price = eq / shares

        vals.append(price)

    return np.array(vals)


if run_button:
    st.subheader("🎲 Monte Carlo (Share Price Distribution)")

    results = run_simulation(ml_fcf)

    mean = np.mean(results)
    p5 = np.percentile(results, 5)
    p95 = np.percentile(results, 95)

    col3, col4, col5 = st.columns(3)
    col3.metric("Mean Price", f"{mean:.2f}")
    col4.metric("Downside (5%)", f"{p5:.2f}")
    col5.metric("Upside (95%)", f"{p95:.2f}")

    fig, ax = plt.subplots()
    ax.hist(results, bins=30)
    ax.axvline(mean)
    ax.axvline(p5)
    ax.axvline(p95)

    st.pyplot(fig)

    st.write("This shows valuation uncertainty under different simulated conditions.")
