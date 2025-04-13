import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Constants
TARIFF = 8.5
CO2_PER_KWH = 0.82

# Streamlit UI
st.set_page_config(page_title="AI-Powered Energy Management Dashboard", layout="wide")
st.title("⚡ AI-Powered Energy Management Dashboard")

# AI Analysis Function
def run_ai_analysis(df):
    df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
    df["Cost_INR"] = df["Energy_kWh"] * TARIFF
    df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH

    X = df[["Month_Num", "Temp_Delta", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
    y = df["Energy_kWh"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    df["Predicted_Energy"] = model.predict(X)
    df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF
    df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

    monthly_avg = df["Energy_kWh"].mean()
    df["Benchmarking"] = df["Energy_kWh"].apply(lambda x: "Above Average" if x > monthly_avg else "Below Average")

    def get_recommendation(row):
        recommendations = []
        reduction_percent = 20
        energy_base = row["Energy_kWh"]

        def savings(est_usage_percent):
            reduction = est_usage_percent * (reduction_percent / 100)
            energy_saving_kwh = energy_base * (reduction / 100)
            cost_saving = energy_saving_kwh * TARIFF
            co2_saving = energy_saving_kwh * CO2_PER_KWH
            return cost_saving, co2_saving

        if row["HVAC_%"] > 50:
            cost, co2 = savings(row["HVAC_%"])
            recommendations.append(f"Optimize HVAC (Save ₹{cost:.0f}, {co2:.1f} kg CO₂/month)")
        if row["Lighting_%"] > 50:
            cost, co2 = savings(row["Lighting_%"])
            recommendations.append(f"Switch to efficient lighting (Save ₹{cost:.0f}, {co2:.1f} kg CO₂/month)")
        if row["Machinery_%"] > 50:
            cost, co2 = savings(row["Machinery_%"])
            recommendations.append(f"Improve machinery scheduling (Save ₹{cost:.0f}, {co2:.1f} kg CO₂/month)")
        if row["Efficiency_Score"] < 85:
            recommendations.append("Consider energy audits for process optimization")

        return "; ".join(recommendations) if recommendations else "All systems operating efficiently"

    df["Smart_Recommendation"] = df.apply(get_recommendation, axis=1)

    return df

# Dashboard Visualization Function
def show_dashboard(df):
    peak_month = df.loc[df["Energy_kWh"].idxmax(), "Month"]
    monthly_avg = df["Energy_kWh"].mean()

    st.markdown("### 🧠 AI Recommendations")
    for i, row in df.iterrows():
        st.markdown(f"""
        <div style='border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 8px; background-color: #f9f9f9;'>
            <b>{row['Month']}</b>: {row['Smart_Recommendation']}<br>
            <i>Efficiency Score: {row['Efficiency_Score']}%</i>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔋 Energy Consumption (Actual vs Predicted)")
        st.line_chart(df[["Month", "Energy_kWh", "Predicted_Energy"]].set_index("Month"))
    with col2:
        st.markdown("#### 💰 Energy Cost (Actual vs Predicted)")
        st.line_chart(df[["Month", "Cost_INR", "Predicted_Cost"]].set_index("Month"))

    st.markdown("### ⚙️ Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📊 Avg Monthly Energy", f"{monthly_avg:.2f} kWh")
    kpi2.metric("💸 Avg Monthly Cost", f"₹{df['Cost_INR'].mean():.2f}")
    kpi3.metric("🌿 Total CO₂ Emitted", f"{df['CO2_kg'].sum():.2f} kg")
    kpi4.metric("🔺 Peak Load Month", f"{peak_month}")

# Input Section
st.sidebar.header("Upload or Enter Data")
input_data = []
uploaded_file = st.sidebar.file_uploader("Upload your monthly data CSV", type="csv")

# Manual Entry
with st.sidebar.expander("Or Enter Monthly Data Manually"):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for month_num, month_name in enumerate(months, start=1):
        energy_kwh = st.number_input(f"{month_name} Energy (kWh)", min_value=0.0, value=0.0, key=f"energy_{month_num}")
        avg_temp = st.number_input(f"{month_name} Outdoor Avg Temp (°C)", value=30.0, key=f"avg_temp_{month_num}")
        indoor_temp = st.number_input(f"{month_name} Indoor Temp (°C)", value=24.0, key=f"indoor_temp_{month_num}")
        humidity = st.slider(f"{month_name} Humidity (%)", 0, 100, 50, key=f"humidity_{month_num}")
        occupancy = st.slider(f"{month_name} Occupancy (%)", 0, 100, 75, key=f"occupancy_{month_num}")
        hvac_pct = st.slider(f"{month_name} HVAC %", 0, 100, 40, key=f"hvac_{month_num}")
        lighting_pct = st.slider(f"{month_name} Lighting %", 0, 100, 30, key=f"lighting_{month_num}")
        machinery_pct = st.slider(f"{month_name} Machinery %", 0, 100, 30, key=f"machinery_{month_num}")

        if energy_kwh > 0:
            input_data.append([month_name, month_num, energy_kwh, avg_temp, indoor_temp, humidity, occupancy, hvac_pct, lighting_pct, machinery_pct])

# Data Handling
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]
    if all(col in df.columns for col in required_cols):
        if st.button("Run AI Analysis (CSV)"):
            df = run_ai_analysis(df)
            st.success("AI Analysis from CSV Complete")
            show_dashboard(df)
    else:
        st.error("Uploaded CSV is missing one or more required columns.")

elif input_data:
    df = pd.DataFrame(input_data, columns=["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"])
    if st.button("Run AI Analysis"):
        df = run_ai_analysis(df)
        st.success("AI Analysis Complete")
        show_dashboard(df)
else:
    st.info("Please upload a CSV file or enter data manually in the sidebar.")
