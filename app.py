import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Valuation Tool", layout="wide")

st.title("📊 Valuation Tool (DCF + Data-Driven + Monte Carlo)")

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("Model Inputs")

years = st.sidebar.slider("Projection Years", 3, 15, 5)

initial_fcf = st.sidebar.number_input("Initial FCF", value=950.0)

growth_rate = st.sidebar.slider("Initial Growth (%)", 0.0, 50.0, 18.0) / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 10.0, 4.5) / 100
wacc = st.sidebar.slider("WACC (%)", 5.0, 20.0, 11.0) / 100

decay = st.sidebar.slider("Growth Decay Factor", 0.7, 0.99, 0.85)

# Capital structure
st.sidebar.subheader("Capital Structure")
debt = st.sidebar.number_input("Debt", value=2000.0)
cash = st.sidebar.number_input("Cash", value=500.0)

# ✅ FIXED: NO MILLIONS
shares = st.sidebar.number_input("Shares Outstanding", value=100.0)

# ML input
st.sidebar.subheader("Historical Cash Flow Input")
fcf_input = st.sidebar.text_input("Enter past values", "450,650,900,1200")

# Monte Carlo
simulations = st.sidebar.slider("Simulation Runs", 100, 3000, 500)
volatility = st.sidebar.slider("Volatility (%)", 1, 30, 12) / 100

run_button = st.sidebar.button("Run Simulation")

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def project_fcf(start_fcf, initial_growth, terminal_growth, years, decay):
    fcf = start_fcf
    fcf_list = []

    for t in range(1, years + 1):
        g_t = terminal_growth + (initial_growth - terminal_growth) * (decay ** t)
        fcf = fcf * (1 + g_t)
        fcf_list.append(fcf)

    return np.array(fcf_list)


def discounted_value(cashflows, wacc):
    return sum(cf / ((1 + wacc) ** (i + 1)) for i, cf in enumerate(cashflows))


def terminal_value(last_fcf, wacc, g, years):
    tv = (last_fcf * (1 + g)) / (wacc - g)  # Excel-consistent
    return tv / ((1 + wacc) ** years)


def equity_value(firm_value, debt, cash):
    return firm_value - debt + cash


# -----------------------------
# DATA-DRIVEN GROWTH
# -----------------------------

try:
    fcf_history = list(map(float, fcf_input.split(",")))
except:
    st.error("Invalid input format")
    st.stop()

growth_rates = [
    (fcf_history[i] - fcf_history[i-1]) / fcf_history[i-1]
    for i in range(1, len(fcf_history))
]

ml_initial_growth = min(np.mean(growth_rates), 0.25)

# -----------------------------
# DCF
# -----------------------------

dcf_fcf = project_fcf(initial_fcf, growth_rate, terminal_growth, years, decay)
dcf_val = discounted_value(dcf_fcf, wacc)
dcf_val += terminal_value(dcf_fcf[-1], wacc, terminal_growth, years)

# -----------------------------
# DATA-DRIVEN MODEL
# -----------------------------

ml_fcf = project_fcf(initial_fcf, ml_initial_growth, terminal_growth, years, decay)
ml_val = discounted_value(ml_fcf, wacc)
ml_val += terminal_value(ml_fcf[-1], wacc, terminal_growth, years)

# -----------------------------
# EQUITY → SHARE PRICE
# -----------------------------

dcf_eq = equity_value(dcf_val, debt, cash)
ml_eq = equity_value(ml_val, debt, cash)

dcf_price = dcf_eq / shares
ml_price = ml_eq / shares

# -----------------------------
# OUTPUT
# -----------------------------

st.subheader("📌 Valuation Results")

col1, col2 = st.columns(2)
col1.metric("DCF Share Price", f"{dcf_price:,.2f}")
col2.metric("Data-Driven Share Price", f"{ml_price:,.2f}")

st.write(f"📊 Data-driven Initial Growth: {ml_initial_growth:.2%}")
st.write(f"📉 Difference: {(dcf_price - ml_price)/dcf_price:.2%}")

# -----------------------------
# CASH FLOWS
# -----------------------------

st.subheader("📈 Projected Cash Flows")

df = pd.DataFrame({
    "Year": range(1, years + 1),
    "DCF FCF": dcf_fcf,
    "Data-driven FCF": ml_fcf
})

st.dataframe(df)

# -----------------------------
# MONTE CARLO (CORRECTED)
# -----------------------------

@st.cache_data
def run_simulation():
    vals = []

    for _ in range(simulations):

        fcf = initial_fcf
        fcf_list = []

        for t in range(1, years + 1):

            noise = np.random.normal(0, volatility)

            g_t = terminal_growth + (ml_initial_growth - terminal_growth) * (decay ** t)
            g_t = g_t * (1 + noise)

            fcf = fcf * (1 + g_t)
            fcf_list.append(fcf)

        val = discounted_value(fcf_list, wacc)
        val += terminal_value(fcf_list[-1], wacc, terminal_growth, years)

        eq = equity_value(val, debt, cash)
        price = eq / shares

        vals.append(price)

    return np.array(vals)


if run_button:
    st.subheader("🎲 Monte Carlo (Share Price)")

    results = run_simulation()

    mean = np.mean(results)
    p5 = np.percentile(results, 5)
    p95 = np.percentile(results, 95)

    col3, col4, col5 = st.columns(3)
    col3.metric("Mean", f"{mean:.2f}")
    col4.metric("Downside (5%)", f"{p5:.2f}")
    col5.metric("Upside (95%)", f"{p95:.2f}")

    fig, ax = plt.subplots()
    ax.hist(results, bins=30)
    ax.axvline(mean)
    ax.axvline(p5)
    ax.axvline(p95)

    ax.set_xlabel("Share Price")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
