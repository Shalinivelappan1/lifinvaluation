import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Fintech Valuation Simulator (DCF + ML + Monte Carlo)")

# -----------------------------
# USER INPUTS
# -----------------------------
st.sidebar.header("Input Parameters")

# General
years = st.sidebar.slider("Projection Years", 3, 10, 5)

# DCF Inputs
initial_fcf = st.sidebar.number_input("Initial FCF", value=1000.0)
growth_rate = st.sidebar.slider("Growth Rate (%)", 0.0, 50.0, 20.0) / 100
wacc = st.sidebar.slider("WACC (%)", 5.0, 20.0, 12.0) / 100
terminal_growth = st.sidebar.slider("Terminal Growth (%)", 0.0, 10.0, 4.0) / 100

# ML Damping
damping = st.sidebar.slider("ML Damping Factor", 0.5, 1.0, 0.8)

# Monte Carlo
simulations = st.sidebar.slider("Monte Carlo Runs", 100, 5000, 1000)

# -----------------------------
# DCF FUNCTION
# -----------------------------
def dcf_valuation(fcf, growth, wacc, terminal_growth, years):
    cashflows = []
    for t in range(1, years + 1):
        fcf = fcf * (1 + growth)
        cashflows.append(fcf / ((1 + wacc) ** t))

    terminal_value = (fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    terminal_discounted = terminal_value / ((1 + wacc) ** years)

    return sum(cashflows) + terminal_discounted

# -----------------------------
# ML-BASED PROJECTION (DAMPED)
# -----------------------------
def ml_projection(fcf, growth, damping, years):
    values = []
    for t in range(1, years + 1):
        adjusted_growth = growth * (damping ** t)
        fcf = fcf * (1 + adjusted_growth)
        values.append(fcf)
    return values

# -----------------------------
# BASE VALUATION
# -----------------------------
dcf_value = dcf_valuation(initial_fcf, growth_rate, wacc, terminal_growth, years)

ml_values = ml_projection(initial_fcf, growth_rate, damping, years)
ml_value = sum([v / ((1 + wacc) ** (i+1)) for i, v in enumerate(ml_values)])

st.subheader("Base Valuation")
st.write(f"DCF Value: {dcf_value:,.2f}")
st.write(f"ML-Based Value: {ml_value:,.2f}")

# -----------------------------
# MONTE CARLO SIMULATION
# -----------------------------
results = []

for _ in range(simulations):
    sim_growth = np.random.normal(growth_rate, 0.05)
    sim_wacc = np.random.normal(wacc, 0.02)
    sim_damping = np.random.normal(damping, 0.05)

    value = dcf_valuation(initial_fcf, sim_growth, sim_wacc, terminal_growth, years)
    results.append(value)

results = np.array(results)

# -----------------------------
# OUTPUT STATISTICS
# -----------------------------
st.subheader("Monte Carlo Results")

mean_val = np.mean(results)
p5 = np.percentile(results, 5)
p95 = np.percentile(results, 95)

st.write(f"Mean Value: {mean_val:,.2f}")
st.write(f"5th Percentile: {p5:,.2f}")
st.write(f"95th Percentile: {p95:,.2f}")

# -----------------------------
# PLOT DISTRIBUTION
# -----------------------------
fig, ax = plt.subplots()
ax.hist(results, bins=50)
ax.set_title("Valuation Distribution")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

st.pyplot(fig)
