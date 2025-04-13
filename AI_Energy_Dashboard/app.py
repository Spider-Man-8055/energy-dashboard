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
template_df = pd.DataFrame(columns=["Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"])

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌱 Smart AI Energy Dashboard for Industries</h1>", unsafe_allow_html=True)
def run_ai_energy_analysis(df):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    # Calculate Temp_Delta
    df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]

    # Define features and target
    features = ["Avg_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%", "Temp_Delta"]
    target = "Energy_kWh"

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df[target])

    # Make predictions
    df["Predicted_Energy"] = model.predict(df[features])
    df["Predicted_Cost"] = df["Predicted_Energy"] * 8.5  # ₹8.5 per unit

    # Efficiency Score
    df["Efficiency_Score"] = 100 - abs(df["Energy_kWh"] - df["Predicted_Energy"]) / df["Energy_kWh"] * 100

    # CO2 Estimation
    df["CO2_Emissions_kg"] = df["Energy_kWh"] * 0.82  # 0.82 kg CO2 per kWh

    # Smart Recommendations
    df["Recommendation"] = df.apply(lambda row: (
        "Optimize HVAC usage" if row["HVAC_%"] > 50 else 
        "Review lighting schedules" if row["Lighting_%"] > 40 else 
        "Machinery load seems high — check for idle time"
    ), axis=1)

    return df
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Run AI analysis
    df = run_ai_energy_analysis(df)

    st.success("✅ AI analysis completed on uploaded data.")
    if st.button("Run Analysis"):
    manual_data = {
        "Month": month,
        "Energy_kWh": energy_kwh,
        "Cost_INR": cost_inr,
        "Avg_Temp": avg_temp,
        "Humidity": humidity,
        "Occupancy_%": occupancy,
        "HVAC_%": hvac,
        "Lighting_%": lighting,
        "Machinery_%": machinery,
        "Indoor_Temp": indoor_temp
    }

    df = pd.DataFrame([manual_data])

    # Run AI analysis
    df = run_ai_energy_analysis(df)

    st.success("✅ AI analysis completed on manual input.")




        st.subheader("📊 Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔋 Energy Consumption (Actual vs Predicted)")
            df_plot_energy = df.copy()
            df_plot_energy["YearMonth"] = df_plot_energy["Year"].astype(str) + "-" + df_plot_energy["Month"]
            df_plot_energy.set_index("YearMonth", inplace=True)
            st.line_chart(df_plot_energy[["Energy_kWh", "Predicted_Energy"]])

        with col2:
            st.markdown("#### 💰 Energy Cost (Actual vs Predicted)")
            df_plot_cost = df.copy()
            df_plot_cost["YearMonth"] = df_plot_cost["Year"].astype(str) + "-" + df_plot_cost["Month"]
            df_plot_cost.set_index("YearMonth", inplace=True)
            st.line_chart(df_plot_cost[["Cost_INR", "Predicted_Cost"]])

        st.markdown("### ⚙️ Key Performance Indicators (KPIs)")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("📊 Avg Monthly Energy", f"{monthly_avg:.2f} kWh")
        kpi2.metric("💸 Avg Monthly Cost", f"₹{df['Cost_INR'].mean():.2f}")
        kpi3.metric("🌿 Total CO₂ Emitted", f"{df['CO2_kg'].sum():.2f} kg")
        kpi4.metric("🔺 Peak Load Month", f"{peak_month}")

        st.markdown("#### ⚡️ Monthly Efficiency Score")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = df["Efficiency_Score"].apply(lambda x: 'green' if x > 100 else ('orange' if x >= 90 else 'red'))
        ax.bar(df["Month"], df["Efficiency_Score"], color=colors)
        ax.axhline(100, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel("Efficiency Score (%)")
        ax.set_title("Monthly Energy Efficiency")
        st.pyplot(fig)

        st.subheader("AI Summary Report")
        st.markdown(f"**Peak Load Month:** {peak_month}")
        st.markdown(f"**Average Energy Usage:** {monthly_avg:.2f} kWh")
        st.markdown(f"**Average Monthly Cost:** ₹{df['Cost_INR'].mean():.2f}")
        st.markdown(f"**Total CO₂ Emissions (kg):** {df['CO2_kg'].sum():.2f}")
        st.markdown(f"**Highest Efficiency Score:** {df['Efficiency_Score'].max():.2f}")

        low_eff_months = df[df["Efficiency_Score"] < 90]["Month"].tolist()
        if low_eff_months:
            st.markdown(f"**Months with Low Efficiency (<90%)**: {', '.join(low_eff_months)}")
        else:
            st.markdown("**All months have good energy efficiency.**")

        inefficient_df = df[df["Efficiency_Score"] < 90]
        inefficient_df["Energy_Saved_kWh"] = inefficient_df["Energy_kWh"] - inefficient_df["Predicted_Energy"]
        inefficient_df["Cost_Saved_INR"] = inefficient_df["Energy_Saved_kWh"] * TARIFF
        inefficient_df["CO2_Saved_kg"] = inefficient_df["Energy_Saved_kWh"] * CO2_PER_KWH

        total_energy_saved = inefficient_df["Energy_Saved_kWh"].sum()
        total_cost_saved = inefficient_df["Cost_Saved_INR"].sum()
        total_co2_saved = inefficient_df["CO2_Saved_kg"].sum()

        if not inefficient_df.empty:
            st.markdown("### Potential Savings if Low-Efficiency Months Improved to 100%")
            st.markdown(f"**Estimated Annual Cost Savings:** ₹{total_cost_saved:.0f}")
            st.markdown(f"**Estimated Annual CO₂ Reduction:** {total_co2_saved:.1f} kg")
            st.markdown(f"**Top Inefficiency Area:** {top_ineff}")
            st.markdown(f"- **Energy Saved:** {total_energy_saved:.2f} kWh")
            st.markdown(f"- **Cost Saved:** ₹{total_cost_saved:.2f}")
            st.markdown(f"- **CO₂ Emissions Avoided:** {total_co2_saved:.2f} kg")

        st.download_button("Download Report as CSV", data=df.to_csv(index=False), file_name="energy_ai_analysis.csv", mime="text/csv")
