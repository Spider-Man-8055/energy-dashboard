import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AI Energy Optimizer Dashboard", layout="wide")
st.title("AI-Powered Commercial Energy Optimizer")

st.markdown("---")

st.header("Step 1: Enter Monthly Energy Data")

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
data = []

for month in months:
    st.subheader(f"{month} Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        energy_kwh = st.number_input(f"{month} - Energy Consumed (kWh)", min_value=0.0, key=f"kwh_{month}")
        avg_temp = st.number_input(f"{month} - Avg Temp (°C)", key=f"temp_{month}")
    with col2:
        humidity = st.number_input(f"{month} - Humidity (%)", key=f"humidity_{month}")
        occupancy = st.slider(f"{month} - Occupancy (%)", 0, 100, key=f"occupancy_{month}")
    with col3:
        hvac = st.slider(f"{month} - HVAC Usage (%)", 0, 100, key=f"hvac_{month}")
        lighting = st.slider(f"{month} - Lighting Usage (%)", 0, 100, key=f"lighting_{month}")
        output = st.number_input(f"{month} - Production Output (optional)", min_value=0.0, key=f"output_{month}")

    cost_inr = round(energy_kwh * 8.5, 2)

    data.append({
        "Month": month,
        "Energy_kWh": energy_kwh,
        "Cost_INR": cost_inr,
        "Avg_Temp": avg_temp,
        "Humidity": humidity,
        "Occupancy_%": occupancy,
        "HVAC_Usage_%": hvac,
        "Lighting_Usage_%": lighting,
        "Production_Output": output,
        "Month_Num": months.index(month)+1
    })

if st.button("Run AI Optimization"):
    df = pd.DataFrame(data)

    st.subheader("Uploaded Monthly Data")
    st.dataframe(df.style.format("{:.2f}"))

    # Train AI Model
    X = df[["Month_Num", "Avg_Temp", "Humidity", "Occupancy_%", "HVAC_Usage_%", "Lighting_Usage_%"]]
    y = df["Energy_kWh"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    df["Predicted_Energy_kWh"] = model.predict(X)
    df["Efficiency (kWh/Output)"] = df.apply(lambda row: row["Energy_kWh"] / row["Production_Output"] if row["Production_Output"] > 0 else np.nan, axis=1)

    st.subheader("AI Model Predictions")
    st.dataframe(df[["Month", "Energy_kWh", "Predicted_Energy_kWh", "Cost_INR", "Efficiency (kWh/Output)"]].style.format("{:.2f}"))

    st.subheader("AI Energy Optimization Suggestions")
    for idx, row in df.iterrows():
        if row["Predicted_Energy_kWh"] > row["Energy_kWh"] * 1.1:
            st.warning(f"{row['Month']}: High predicted energy! Consider reducing HVAC or lighting load.")
        elif row["Efficiency (kWh/Output)"] and row["Efficiency (kWh/Output)"] > 1.5 * df["Efficiency (kWh/Output)"].mean():
            st.info(f"{row['Month']}: Low production efficiency detected. Consider reviewing equipment usage or shift timings.")
        else:
            st.success(f"{row['Month']}: Energy usage is within optimized range.")
