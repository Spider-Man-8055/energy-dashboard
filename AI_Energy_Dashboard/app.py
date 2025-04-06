import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Constants
TARIFF = 8.5  # INR per kWh
CO2_PER_KWH = 0.82  # kg CO2 per kWh (India average)

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.title("AI-Powered Energy Management for Industrial Facility")
# CSV Template Download
import io

# CSV Template + Upload + Manual Entry
st.markdown("### Option 1: Upload Monthly Energy Data (CSV)")

# Template download
template_df = pd.DataFrame({
    "Month": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    "Month_Num": list(range(1, 13)),
    "Energy_kWh": [0.0] * 12,
    "Avg_Temp": [25.0] * 12,
    "Humidity": [50.0] * 12,
    "Occupancy_%": [75] * 12,
    "HVAC_%": [40] * 12,
    "Lighting_%": [30] * 12,
    "Machinery_%": [30] * 12
})
csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)
st.download_button("Download CSV Template", data=csv_buffer.getvalue(), file_name="energy_input_template.csv", mime="text/csv")

# Upload CSV
uploaded_file = st.file_uploader("Upload your completed CSV file", type=["csv"])
# Expected columns in uploaded CSV
REQUIRED_COLUMNS = ["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Humidity", 
                    "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]

uploaded_file = st.file_uploader("Upload CSV file with energy data", type=["csv"])

df = None  # Initialize DataFrame

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    
    # Check for missing columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_uploaded.columns]
    
    if missing_cols:
        st.error(f"Uploaded CSV is missing the following required column(s): {', '.join(missing_cols)}")
        st.stop()
    else:
        df = df_uploaded.copy()
        st.success("✅ CSV uploaded and validated successfully.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV Uploaded Successfully!")
    st.dataframe(df)
else:
    st.markdown("### Option 2: Enter data manually below")
    
    # Manual data entry
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    input_data = []

    for i in range(12):
       with st.expander(f"Month: {months[i]}"):
    energy = st.number_input(f"Energy used (kWh) - {months[i]}", min_value=0.0, value=0.0, key=f"e_{i}")
    avg_temp = st.number_input(f"Outdoor Temp (°C) - {months[i]}", value=30.0, key=f"t_{i}")
    indoor_temp = st.number_input(f"Indoor Temp (°C) - {months[i]}", value=24.0, key=f"indoor_{i}")
    humidity = st.number_input(f"Humidity (%) - {months[i]}", value=50.0, key=f"h_{i}")
    occupancy = st.slider(f"Occupancy (%) - {months[i]}", 0, 100, 75, key=f"o_{i}")
    
    hvac = st.slider(f"HVAC Usage (%) - {months[i]}", 0, 100, 40, key=f"hvac_{i}")
    lighting = st.slider(f"Lighting Usage (%) - {months[i]}", 0, 100, 30, key=f"light_{i}")
    machinery = st.slider(f"Machinery Usage (%) - {months[i]}", 0, 100, 30, key=f"mach_{i}")
    
    # Live cost estimate
    month_cost = energy * TARIFF
    st.markdown(f"**Estimated Monthly Cost:** ₹{month_cost:.2f}")

    # Live usage breakdown chart
    usage_pie = pd.DataFrame({
        'Category': ['HVAC', 'Lighting', 'Machinery'],
        'Usage %': [hvac, lighting, machinery]
    })
    st.markdown("**Usage Distribution**")
    st.pyplot(plt.figure(figsize=(3.5, 3.5)))
    plt.pie(usage_pie["Usage %"], labels=usage_pie["Category"], autopct='%1.1f%%', startangle=90)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Append new indoor_temp to data
    input_data.append([months[i], i+1, energy, avg_temp, indoor_temp, humidity, occupancy, hvac, lighting, machinery])

    df = pd.DataFrame(input_data, columns=["Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"])
df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
df["Cost_INR"] = df["Energy_kWh"] * TARIFF
df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH

if st.button("Run AI Analysis"):
    st.success("AI Analysis Complete")
    
    # Features and target
    X = df[["Month_Num", "Temp_Delta", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
    y = df["Energy_kWh"]

    # Train simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predictions
    df["Predicted_Energy"] = model.predict(X)
    df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF

    # Efficiency Score
    avg_pred = df["Predicted_Energy"].mean()
    df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

    # Peak Load Detection
    peak_month = df.loc[df["Energy_kWh"].idxmax(), "Month"]

    # Benchmarking
    monthly_avg = df["Energy_kWh"].mean()
    df["Benchmarking"] = df["Energy_kWh"].apply(lambda x: "Above Average" if x > monthly_avg else "Below Average")

    # Smart Recommendations
    def get_recommendation(row):
    recommendations = []

    # Constants for potential savings
    reduction_percent = 20  # Assume 20% reduction possible
    energy_base = row["Energy_kWh"]

    def savings(est_usage_percent):
        reduction = est_usage_percent * (reduction_percent / 100)
        energy_saving_kwh = energy_base * (reduction / 100)
        cost_saving = energy_saving_kwh * TARIFF
        co2_saving = energy_saving_kwh * CO2_PER_KWH
        return cost_saving, co2_saving

    # HVAC usage high
    if row["HVAC_%"] > 50:
        cost, co2 = savings(row["HVAC_%"])
        recommendations.append(f"Optimize HVAC (Save ₹{cost:.0f}, {co2:.1f} kg CO₂/month)")

    # Lighting usage high
    if row["Lighting_%"] > 50:
        cost, co2 = savings(row["Lighting_%"])
        recommendations.append(f"Switch to efficient lighting (Save ₹{cost:.0f}, {co2:.1f} kg CO₂/month)")

    # Machinery usage high
    if row["Machinery_%"] > 50:
        cost, co2 = savings(row["Machinery_%"])
        recommendations.append(f"Improve machinery scheduling (Save ₹{cost:.0f}, {co2:.1f} kg CO₂/month)")

    # Efficiency low
    if row["Efficiency_Score"] < 85:
        recommendations.append("Consider energy audits for process optimization")

    if not recommendations:
        return "All systems operating efficiently"

    return "; ".join(recommendations)

    
    df["Smart_Recommendation"] = df.apply(get_recommendation, axis=1)
# ---- Annual Savings Summary ----

# Helper to extract numbers from recommendation strings
def extract_savings(text, value_type="cost"):
    import re
    matches = re.findall(r"₹(\d+)|([\d.]+) kg", text)
    cost_saving = 0
    co2_saving = 0
    for cost, co2 in matches:
        if cost:
            cost_saving += float(cost)
        if co2:
            co2_saving += float(co2)
    return cost_saving if value_type == "cost" else co2_saving

df["Est_Cost_Saved"] = df["Smart_Recommendation"].apply(lambda x: extract_savings(x, "cost"))
df["Est_CO2_Saved"] = df["Smart_Recommendation"].apply(lambda x: extract_savings(x, "co2"))

total_cost_saved = df["Est_Cost_Saved"].sum()
total_co2_saved = df["Est_CO2_Saved"].sum()

# Identify top inefficiency categories
ineff_areas = {
    "HVAC": df[df["HVAC_%"] > 50].shape[0],
    "Lighting": df[df["Lighting_%"] > 50].shape[0],
    "Machinery": df[df["Machinery_%"] > 50].shape[0]
}
top_ineff = max(ineff_areas, key=ineff_areas.get)

    # Display output
    st.dataframe(df)

    # Charts
    st.subheader("Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df[["Month", "Energy_kWh"]].set_index("Month"))
        st.bar_chart(df[["Month", "Predicted_Energy"]].set_index("Month"))

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = df["Efficiency_Score"].apply(lambda x: 'green' if x > 100 else ('orange' if x >= 90 else 'red'))
        ax.bar(df["Month"], df["Efficiency_Score"], color=colors)
        ax.axhline(100, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel("Efficiency Score (%)")
        ax.set_title("Monthly Energy Efficiency")
        st.pyplot(fig)
        st.line_chart(df[["Month", "Cost_INR", "Predicted_Cost"]].set_index("Month"))

    # Summary
    st.subheader("AI Summary Report")
    st.markdown(f"**Peak Load Month:** {peak_month}")
    st.markdown(f"**Average Energy Usage:** {monthly_avg:.2f} kWh")
    st.markdown(f"**Average Monthly Cost:** ₹{df['Cost_INR'].mean():.2f}")
    st.markdown(f"**Total CO₂ Emissions (kg):** {df['CO2_kg'].sum():.2f}")
    st.markdown(f"**Highest Efficiency Score:** {df['Efficiency_Score'].max():.2f}")
          # Identify months with low efficiency
low_eff_months = df[df["Efficiency_Score"] < 90]["Month"].tolist()

# Show flagged months
if low_eff_months:
    st.markdown(f"**Months with Low Efficiency (<90%)**: {', '.join(low_eff_months)}")
else:
    st.markdown("**All months have good energy efficiency.**")
# Potential savings from inefficient months
inefficient_df = df[df["Efficiency_Score"] < 90]
inefficient_df["Energy_Saved_kWh"] = inefficient_df["Energy_kWh"] - inefficient_df["Predicted_Energy"]
inefficient_df["Cost_Saved_INR"] = inefficient_df["Energy_Saved_kWh"] * TARIFF
inefficient_df["CO2_Saved_kg"] = inefficient_df["Energy_Saved_kWh"] * CO2_PER_KWH

total_energy_saved = inefficient_df["Energy_Saved_kWh"].sum()
total_cost_saved = inefficient_df["Cost_Saved_INR"].sum()
total_co2_saved = inefficient_df["CO2_Saved_kg"].sum()

# Show insights
if not inefficient_df.empty:
    st.markdown("### Potential Savings if Low-Efficiency Months Improved to 100%")
    st.markdown(f"**Estimated Annual Cost Savings:** ₹{total_cost_saved:.0f}")
    st.markdown(f"**Estimated Annual CO₂ Reduction:** {total_co2_saved:.1f} kg")
    st.markdown(f"**Top Inefficiency Area:** {top_ineff}")
    st.markdown(f"- **Energy Saved:** {total_energy_saved:.2f} kWh")
    st.markdown(f"- **Cost Saved:** ₹{total_cost_saved:.2f}")
    st.markdown(f"- **CO₂ Emissions Avoided:** {total_co2_saved:.2f} kg")


    st.download_button("Download Report as CSV", data=df.to_csv(index=False), file_name="energy_ai_analysis.csv", mime="text/csv")
