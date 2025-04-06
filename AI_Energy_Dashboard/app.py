import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Constants
TARIFF = 8.5  # INR per kWh
CO2_PER_KWH = 0.82  # kg CO2 per kWh (India average)

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.title("AI-Powered Energy Management for Industrial Facility")

st.markdown("### Enter monthly data manually for analysis")

# Data entry table for 12 months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
input_data = []

for i in range(12):
    with st.expander(f"Month: {months[i]}"):
        energy = st.number_input(f"Energy used (kWh) - {months[i]}", min_value=0.0, key=f"e_{i}")
        avg_temp = st.number_input(f"Average Temp (°C) - {months[i]}", key=f"t_{i}")
        humidity = st.number_input(f"Humidity (%) - {months[i]}", key=f"h_{i}")
        occupancy = st.slider(f"Occupancy (%) - {months[i]}", 0, 100, 75, key=f"o_{i}")
        hvac = st.slider(f"HVAC Usage (%) - {months[i]}", 0, 100, 40, key=f"hvac_{i}")
        lighting = st.slider(f"Lighting Usage (%) - {months[i]}", 0, 100, 30, key=f"light_{i}")
        machinery = st.slider(f"Machinery Usage (%) - {months[i]}", 0, 100, 30, key=f"mach_{i}")

        input_data.append([months[i], i+1, energy, avg_temp, humidity, occupancy, hvac, lighting, machinery])

df = pd.DataFrame(input_data, columns=["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"])
df["Cost_INR"] = df["Energy_kWh"] * TARIFF
df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH

if st.button("Run AI Analysis"):
    st.success("AI Analysis Complete")
    
    # Features and target
    X = df[["Month_Num", "Avg_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
    y = df["Energy_kWh"]

    # Train simple model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    df["Predicted_Energy"] = model.predict(X)
    df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF

    # Efficiency Score
    avg_pred = df["Predicted_Energy"].mean()
    df["Efficiency_Score"] = np.round((avg_pred / df["Predicted_Energy"]) * 100, 2)

    # Peak Load Detection
    peak_month = df.loc[df["Energy_kWh"].idxmax(), "Month"]

    # Benchmarking
    monthly_avg = df["Energy_kWh"].mean()
    df["Benchmarking"] = df["Energy_kWh"].apply(lambda x: "Above Average" if x > monthly_avg else "Below Average")

    # Smart Recommendations
    def get_recommendation(row):
        if row["HVAC_%"] > 50:
            return "Optimize HVAC systems"
        elif row["Lighting_%"] > 50:
            return "Use efficient lighting"
        elif row["Machinery_%"] > 50:
            return "Schedule machinery better"
        elif row["Efficiency_Score"] < 85:
            return "Improve energy practices"
        else:
            return "Good"
    
    df["Smart_Recommendation"] = df.apply(get_recommendation, axis=1)

    # Display output
    st.dataframe(df)

    # Charts
    st.subheader("Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df[["Month", "Energy_kWh"]].set_index("Month"))
        st.bar_chart(df[["Month", "Predicted_Energy"]].set_index("Month"))

    with col2:
        st.line_chart(df[["Month", "Efficiency_Score"]].set_index("Month"))
        st.line_chart(df[["Month", "Cost_INR", "Predicted_Cost"]].set_index("Month"))

    # Summary
    st.subheader("AI Summary Report")
    st.markdown(f"**Peak Load Month:** {peak_month}")
    st.markdown(f"**Average Energy Usage:** {monthly_avg:.2f} kWh")
    st.markdown(f"**Average Monthly Cost:** ₹{df['Cost_INR'].mean():.2f}")
    st.markdown(f"**Total CO₂ Emissions (kg):** {df['CO2_kg'].sum():.2f}")
    st.markdown(f"**Highest Efficiency Score:** {df['Efficiency_Score'].max():.2f}")

    st.download_button("Download Report as CSV", data=df.to_csv(index=False), file_name="energy_ai_analysis.csv", mime="text/csv")
