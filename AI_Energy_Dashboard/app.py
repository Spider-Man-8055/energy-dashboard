import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Manual Energy Dashboard", layout="wide")
st.title("Manual Input - AI Energy Optimizer Dashboard")

months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# Collect inputs for each month
data = []
st.subheader("Enter Monthly Energy Data (Manually)")

for i, month in enumerate(months):
    with st.expander(f"{month}"):
        energy = st.number_input(f"{month} - Energy (kWh)", min_value=0.0, step=0.1, key=f"energy_{i}")
        cost = st.number_input(f"{month} - Cost (INR)", min_value=0.0, step=0.1, key=f"cost_{i}")
        temp = st.number_input(f"{month} - Avg Temperature (°C)", min_value=0.0, step=0.1, key=f"temp_{i}")
        humidity = st.number_input(f"{month} - Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, key=f"humidity_{i}")
        occupancy = st.slider(f"{month} - Occupancy (%)", min_value=0, max_value=100, value=80, key=f"occupancy_{i}")
        data.append([month, i+1, energy, cost, temp, humidity, occupancy])

# Create DataFrame
columns = ["Month", "Month_Num", "Energy_kWh", "Cost_INR", "Avg_Temp", "Humidity", "Occupancy_%"]
df = pd.DataFrame(data, columns=columns)

if st.button("Run AI Energy Analysis"):
    st.subheader("Raw Data")
    st.dataframe(df)

    # Model input
    X = df[["Month_Num", "Avg_Temp", "Humidity", "Occupancy_%"]]
    y = df["Energy_kWh"]

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    df["Predicted_Energy_kWh"] = model.predict(X)

    # Metrics
    st.subheader("Model Accuracy (Trained on Input Data)")
    st.write(f"MAE: {mean_absolute_error(y, df['Predicted_Energy_kWh']):.2f}")
    st.write(f"MSE: {mean_squared_error(y, df['Predicted_Energy_kWh']):.2f}")
    st.write(f"R² Score: {r2_score(y, df['Predicted_Energy_kWh']):.2f}")

    # Charts
    st.subheader("Actual vs Predicted Energy Usage")
    st.line_chart(df[["Energy_kWh", "Predicted_Energy_kWh"]])

    # AI Recommendations
    st.subheader("AI-Powered Energy Recommendations")
    for idx, row in df.iterrows():
        suggestions = []
        if row["Occupancy_%"] > 90:
            suggestions.append("Consider optimizing HVAC systems in highly occupied areas.")
        if row["Humidity"] > 70:
            suggestions.append("Dehumidifiers can improve AC efficiency.")
        if row["Avg_Temp"] > 30:
            suggestions.append("Use shading or insulation to reduce cooling load.")
        if not suggestions:
            suggestions.append("Energy performance looks good this month.")
        st.markdown(f"**{row['Month']}**: {' | '.join(suggestions)}")
