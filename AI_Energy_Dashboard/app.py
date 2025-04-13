import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io

# Constants
TARIFF = 8.5
CO2_PER_KWH = 0.82
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
template_df = pd.DataFrame(columns=["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"])

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 Smart AI Energy Dashboard for Industries</h1>", unsafe_allow_html=True)

# CSV Upload
st.markdown("## 📁 Data Input Options")
with st.expander("📤 Upload CSV File"):
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Download CSV Template", data=csv_buffer.getvalue(), file_name="energy_input_template.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload your completed CSV file", type=["csv"])

# Manual Data Entry
with st.expander("📝 Manually Enter Data"):
    st.write("Enter the energy usage and other parameters manually.")
    
    months_input = st.selectbox("Month", months, index=0)
    month_num = months.index(months_input) + 1
    energy_kwh = st.number_input("Energy Consumption (kWh)", min_value=0, step=1)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=-50, step=0.1)
    indoor_temp = st.number_input("Indoor Temperature (°C)", min_value=-50, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
    occupancy = st.number_input("Occupancy (%)", min_value=0, max_value=100, step=1)
    hvac = st.number_input("HVAC Usage (%)", min_value=0, max_value=100, step=1)
    lighting = st.number_input("Lighting Usage (%)", min_value=0, max_value=100, step=1)
    machinery = st.number_input("Machinery Usage (%)", min_value=0, max_value=100, step=1)

    if st.button("Run AI Analysis (Manual Input)"):
        manual_data = {
            "Month": [months_input],
            "Month_Num": [month_num],
            "Energy_kWh": [energy_kwh],
            "Avg_Temp": [avg_temp],
            "Indoor_Temp": [indoor_temp],
            "Humidity": [humidity],
            "Occupancy_%": [occupancy],
            "HVAC_%": [hvac],
            "Lighting_%": [lighting],
            "Machinery_%": [machinery]
        }

        df = pd.DataFrame(manual_data)
        df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
        df["Cost_INR"] = df["Energy_kWh"] * TARIFF
        df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH

        st.success("AI Analysis from Manual Input Complete")

        X = df[["Month_Num", "Temp_Delta", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
        y = df["Energy_kWh"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        df["Predicted_Energy"] = model.predict(X)
        df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF
        df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

        # Recommendations, CO2 savings, and energy savings logic
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

        st.markdown("### 🧠 AI Recommendations")
        st.write(df[["Month", "Smart_Recommendation"]])

        st.markdown("### 📊 Visualization")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔋 Energy Consumption (Actual vs Predicted)")
            st.line_chart(df[["Month", "Energy_kWh", "Predicted_Energy"]].set_index("Month"))
        with col2:
            st.markdown("#### 💰 Energy Cost (Actual vs Predicted)")
            st.line_chart(df[["Month", "Cost_INR", "Predicted_Cost"]].set_index("Month"))

        st.markdown("### ⚙️ Key Performance Indicators (KPIs)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("📊 Avg Monthly Energy", f"{df['Energy_kWh'].mean():.2f} kWh")
        kpi2.metric("💸 Avg Monthly Cost", f"₹{df['Cost_INR'].mean():.2f}")
        kpi3.metric("🌿 Total CO₂ Emissions", f"{df['CO2_kg'].sum():.2f} kg")
        kpi4.metric("🔺 Efficiency Score", f"{df['Efficiency_Score'].max():.2f} %")

# Handle CSV file upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]
    df = df.fillna(0)  # Fill missing data with zero

    if all(col in df.columns for col in required_cols):
        df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
        df["Cost_INR"] = df["Energy_kWh"] * TARIFF
        df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH

        if st.button("Run AI Analysis (CSV)"):
            st.success("AI Analysis from CSV Complete")

            X = df[["Month_Num", "Temp_Delta", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
            y = df["Energy_kWh"]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            df["Predicted_Energy"] = model.predict(X)
            df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF
            df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

            st.markdown("### 🧠 AI Recommendations")
            for i, row in df.iterrows():
                st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 8px; background-color: #f9f9f9;'>"
                            f"<b>{row['Month']}</b>: {row['Smart_Recommendation']}<br><i>Efficiency Score: {row['Efficiency_Score']}%</i></div>", unsafe_allow_html=True)

            # Visualization and KPIs
            st.markdown("### 📊 Visualization")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔋 Energy Consumption (Actual vs Predicted)")
                st.line_chart(df[["Month", "Energy_kWh", "Predicted_Energy"]].set_index("Month"))
            with col2:
                st.markdown("#### 💰 Energy Cost (Actual vs Predicted)")
                st.line_chart(df[["Month", "Cost_INR", "Predicted_Cost"]].set_index("Month"))
    else:
        st.error("Missing or incorrect columns in the uploaded CSV file.")
