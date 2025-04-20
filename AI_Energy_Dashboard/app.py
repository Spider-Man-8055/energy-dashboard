import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io

# ─────────────── Constants & Template ───────────────
TARIFF = 8.5
CO2_PER_KWH = 0.82
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

template_df = pd.DataFrame(columns=[
    "Year","Month","Month_Num","Energy_kWh","Avg_Temp",
    "Indoor_Temp","Humidity","Occupancy_%","HVAC_%","Lighting_%","Machinery_%"
])

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align:center;color:#2E8B57;'>🌱 Smart AI Energy Dashboard for Industries</h1>",
    unsafe_allow_html=True
)

# ─────────────── Analysis Function ───────────────
def run_ai_energy_analysis(df):
    df = df.copy()
    # ← Add actual cost so we can plot it later
    df["Cost_INR"] = df["Energy_kWh"] * TARIFF

    # Feature engineering
    df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]

    # Train model
    X = df[["Year","Month_Num","Temp_Delta","Humidity",
            "Occupancy_%","HVAC_%","Lighting_%","Machinery_%"]]
    y = df["Energy_kWh"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict & score
    df["Predicted_Energy"] = model.predict(X)
    df["Predicted_Cost"]   = df["Predicted_Energy"] * TARIFF
    df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

    # Benchmarks
    peak_month  = df.loc[df["Energy_kWh"].idxmax(), "Month"]
    monthly_avg = df["Energy_kWh"].mean()
    df["Benchmarking"] = df["Energy_kWh"].apply(
        lambda x: "Above Average" if x > monthly_avg else "Below Average"
    )

    # Top inefficiency area
    ineff_counts = {
        "HVAC":      (df["HVAC_%"]      > 50).sum(),
        "Lighting":  (df["Lighting_%"]  > 50).sum(),
        "Machinery": (df["Machinery_%"] > 50).sum(),
    }
    top_ineff = max(ineff_counts, key=ineff_counts.get)

    # Build recommendations + numeric savings
    def compute_row(row):
        recs = []
        cost_sum, co2_sum = 0.0, 0.0
        base = row["Energy_kWh"]
        def savings(pct):
            saved_kwh = base * (pct * 0.20) / 100
            return saved_kwh * TARIFF, saved_kwh * CO2_PER_KWH

        if row["HVAC_%"] > 50:
            c, c2 = savings(row["HVAC_%"])
            recs.append(f"Optimize HVAC (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
            cost_sum += c; co2_sum += c2
        if row["Lighting_%"] > 50:
            c, c2 = savings(row["Lighting_%"])
            recs.append(f"Efficient lighting (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
            cost_sum += c; co2_sum += c2
        if row["Machinery_%"] > 50:
            c, c2 = savings(row["Machinery_%"])
            recs.append(f"Improve machinery (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
            cost_sum += c; co2_sum += c2
        if row["Efficiency_Score"] < 85:
            recs.append("Consider energy audits")

        text = "; ".join(recs) if recs else "All systems efficient"
        return pd.Series([text, cost_sum, co2_sum],
                         index=["Smart_Recommendation","Est_Cost_Saved","Est_CO2_Saved"])

    rec_df = df.apply(compute_row, axis=1)
    df = pd.concat([df, rec_df], axis=1)

    total_cost_saved = df["Est_Cost_Saved"].sum()
    total_co2_saved  = df["Est_CO2_Saved"].sum()

    # Display per‐month recs
    st.markdown("### 🧠 AI Recommendations")
    for _, r in df.iterrows():
        st.markdown(
            f"""<div style='border:1px solid #ddd; padding:10px; margin:5px 0; border-radius:8px; background:#f9f9f9;'>
                 <b>{r['Month']}</b>: {r['Smart_Recommendation']}<br>
                 <i>Efficiency Score: {r['Efficiency_Score']}%</i>
               </div>""",
            unsafe_allow_html=True
        )

    return df, monthly_avg, peak_month, top_ineff, total_cost_saved, total_co2_saved

# ─────────────── CSV Upload Section ───────────────
st.markdown("## 📁 Data Input Options")
with st.expander("📄 Upload CSV File"):
    buf = io.StringIO()
    template_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        data=buf.getvalue(),
        file_name="energy_input_template.csv",
        mime="text/csv"
    )
    uploaded = st.file_uploader("Upload your completed CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)

# ─────────────── Manual Entry Section ───────────────
st.markdown("### OR")
st.markdown("## ✍️ Manual Entry")

years    = list(range(2000, 2031))
sel_year = st.selectbox("Select Year", years)
rows     = []

for i, m in enumerate(months):
    with st.expander(f"Month: {m}"):
        energy       = st.number_input(f"Energy (kWh) - {m}", value=0.0, key=f"e_{i}")
        outdoor_temp = st.number_input(f"Outdoor Temp (°C) - {m}", value=30.0, key=f"at_{i}")
        indoor_temp  = st.number_input(f"Indoor Temp (°C) - {m}",  value=24.0, key=f"it_{i}")
        humidity     = st.number_input(f"Humidity (%) - {m}",     value=50.0, key=f"h_{i}")
        occupancy    = st.slider(f"Occupancy (%) - {m}", 0, 100, 75, key=f"oc_{i}")
        hvac         = st.slider(f"HVAC Usage (%) - {m}", 0, 100, 40, key=f"hv_{i}")
        lighting     = st.slider(f"Lighting Usage (%) - {m}", 0, 100, 30, key=f"li_{i}")
        machinery    = st.slider(f"Machinery Usage (%) - {m}", 0, 100, 30, key=f"ma_{i}")

        rows.append([
            sel_year, m, i+1,
            energy, outdoor_temp, indoor_temp,
            humidity, occupancy, hvac,
            lighting, machinery
        ])

if rows:
    df = pd.DataFrame(rows, columns=[
        "Year","Month","Month_Num","Energy_kWh","Avg_Temp",
        "Indoor_Temp","Humidity","Occupancy_%","HVAC_%","Lighting_%","Machinery_%"
    ])

# ─────────────── Run Analysis Buttons ───────────────
if st.button("💡 Run AI Analysis on Uploaded Data") and "df" in locals():
    (st.session_state.df_analyzed,
     st.session_state.avg,
     st.session_state.peak,
     st.session_state.top_ineff,
     st.session_state.cost_saved,
     st.session_state.co2_saved) = run_ai_energy_analysis(df)
    st.success("✅ AI analysis completed on uploaded data.")

if st.button("💡 Run AI Analysis on Manual Entry") and "df" in locals():
    (st.session_state.df_analyzed,
     st.session_state.avg,
     st.session_state.peak,
     st.session_state.top_ineff,
     st.session_state.cost_saved,
     st.session_state.co2_saved) = run_ai_energy_analysis(df)
    st.success("✅ AI analysis completed on manual input.")

# ─────────────── Persisted Visualization, KPIs & Report ───────────────
if "df_analyzed" in st.session_state:
    df2 = st.session_state.df_analyzed

    # Visualization
    st.subheader("📊 Visualization")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔋 Energy (Actual vs Predicted)")
        df2_plot = df2.set_index(df2["Year"].astype(str)+"-"+df2["Month"])
        st.line_chart(df2_plot[["Energy_kWh","Predicted_Energy"]])
    with c2:
        st.markdown("#### 💰 Cost (Actual vs Predicted)")
        st.line_chart(df2_plot[["Cost_INR","Predicted_Cost"]])

    # KPIs
    st.markdown("### ⚙️ Key Performance Indicators")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Avg Monthly Energy", f"{df2['Energy_kWh'].mean():.2f} kWh")
    k2.metric("Avg Monthly Cost",   f"₹{df2['Predicted_Cost'].mean():.2f}")
    k3.metric("Total CO₂ Emitted",   f"{(df2['Energy_kWh']*CO2_PER_KWH).sum():.2f} kg")
    k4.metric("🔺 Peak Load Month",    st.session_state.peak)

    # Efficiency Chart
    st.markdown("#### ⚡️ Monthly Efficiency Score")
    fig, ax = plt.subplots(figsize=(8,4))
    colors = df2["Efficiency_Score"].apply(lambda x: "green" if x>100 else ("orange" if x>=90 else "red"))
    ax.bar(df2["Month"], df2["Efficiency_Score"], color=colors)
    ax.axhline(100, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Efficiency Score (%)")
    ax.set_title("Monthly Energy Efficiency")
    st.pyplot(fig)

    # Summary Report
    st.subheader("AI Summary Report")
    st.markdown(f"**Peak Load Month:** {st.session_state.peak}")
    st.markdown(f"**Average Energy Usage:** {st.session_state.avg:.2f} kWh")
    st.markdown(f"**Average Monthly Cost:** ₹{df2['Predicted_Cost'].mean():.2f}")
    st.markdown(f"**Total CO₂ Emissions:** {(df2['Energy_kWh']*CO2_PER_KWH).sum():.2f} kg")
    st.markdown(f"**Highest Efficiency Score:** {df2['Efficiency_Score'].max():.2f}")

    low = df2[df2["Efficiency_Score"]<90]["Month"].tolist()
    if low:
        st.markdown(f"**Months with Low Efficiency (<90%):** {', '.join(low)}")
    else:
        st.markdown("**All months have good energy efficiency.**")

    # Potential Savings
    st.markdown("### Potential Savings if Low‑Efficiency Months Hit 100%")
    st.markdown(f"- **Annual Cost Savings:** ₹{st.session_state.cost_saved:.0f}")
    st.markdown(f"- **Annual CO₂ Reduction:** {st.session_state.co2_saved:.1f} kg")
    st.markdown(f"- **Top Inefficiency Area:** {st.session_state.top_ineff}")

    # CSV Download
    st.download_button(
        "Download Report as CSV",
        data=df2.to_csv(index=False),
        file_name="energy_ai_analysis.csv",
        mime="text/csv"
    )
