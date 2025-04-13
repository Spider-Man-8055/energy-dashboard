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
           X = df[["Year", "Month_Num", "Temp_Delta", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
           y = df["Energy_kWh"]

           model = RandomForestRegressor(n_estimators=100, random_state=42)
           model.fit(X, y)

           df["Predicted_Energy"] = model.predict(X)
           df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF
           df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

           peak_month = df.loc[df["Energy_kWh"].idxmax(), "Month"]
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

          ineff_areas = {
                "HVAC": df[df["HVAC_%"] > 50].shape[0],
                "Lighting": df[df["Lighting_%"] > 50].shape[0],
                "Machinery": df[df["Machinery_%"] > 50].shape[0]
            }
          top_ineff = max(ineff_areas, key=ineff_areas.get)

            # === Dashboard Output (Reused from Manual Input Block) ==
          st.markdown("### 🧠 AI Recommendations")
          for i, row in df.iterrows():
                st.markdown(f"""<div style='border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 8px; background-color: #f9f9f9;'>
                    <b>{row['Month']}</b>: {row['Smart_Recommendation']}<br>
                    <i>Efficiency Score: {row['Efficiency_Score']}%</i></div>""", unsafe_allow_html=True)
                      if 'df' in locals() and "Predicted_Energy" in df.columns:          
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
          inefficient_df = inefficient_df.copy()  
          inefficient_df["Energy_Saved_kWh"] = inefficient_df["Energy_kWh"] - inefficient_df["Predicted_Energy"]
          inefficient_df["Cost_Saved_INR"] = inefficient_df["Energy_Saved_kWh"] * TARIFF
          inefficient_df["CO2_Saved_kg"] = inefficient_df["Energy_Saved_kWh"] * CO2_PER_KWH

          total_energy_saved = inefficient_df["Energy_Saved_kWh"].sum()
          estimated_cost_saving = inefficient_df["Cost_Saved_INR"].sum()
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
          return df, monthly_avg, peak_month, top_ineff, total_cost_saved, total_co2_saved
st.markdown("## 📁 Data Input Options")
with st.expander("📤 Upload CSV File"):
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Download CSV Template", data=csv_buffer.getvalue(), file_name="energy_input_template.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload your completed CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required_cols = ["Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]
    if all(col in df.columns for col in required_cols):
        df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
        df["Cost_INR"] = df["Energy_kWh"] * TARIFF
        df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH
if st.button("💡 Run AI Energy Analysis"):
    if 'df' in locals() and not df.empty:
    # Run AI analysis
        df, monthly_avg, peak_month, top_ineff, total_cost_saved, total_co2_saved  = run_ai_energy_analysis(df)
        st.success("✅ AI analysis completed on uploaded data.")
          #manually uploading section
    else:
        st.error("Uploaded CSV is missing one or more required columns.")

st.markdown("### OR")
st.markdown("## ✍️ Manual Entry")
years = list(range(2000, 2031))  # For example, let users select years from 2000 to 2030
selected_year = st.selectbox("Select Year", years)
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

        month_cost = energy * TARIFF
        st.markdown(f"**Estimated Monthly Cost:** ₹{month_cost:.2f}")

        usage_pie = pd.DataFrame({
            'Category': ['HVAC', 'Lighting', 'Machinery'],
            'Usage %': [hvac, lighting, machinery]
        })

        st.markdown("**Usage Distribution**")
        fig1, ax1 = plt.subplots()
        ax1.pie(usage_pie["Usage %"], labels=usage_pie["Category"], autopct='%1.1f%%', startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        input_data.append([selected_year, months[i], i+1, energy, avg_temp, indoor_temp, humidity, occupancy, hvac, lighting, machinery])

if input_data:
    df = pd.DataFrame(input_data, columns=["Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp", "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"])
    df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
    df["Cost_INR"] = df["Energy_kWh"] * TARIFF
    df["CO2_kg"] = df["Energy_kWh"] * CO2_PER_KWH

if st.button("Run AI Analysis"):
    # Run AI analysis
    df, monthly_avg, peak_month, top_ineff, total_cost_saved, total_co2_saved = run_ai_energy_analysis(df)
    st.success("✅ AI analysis completed on manual input.")

