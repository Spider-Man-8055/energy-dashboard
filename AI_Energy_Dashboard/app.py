import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io

# ─────────────── Constants & Template ───────────────
TARIFF = 8.5
CO2_PER_KWH = 0.82
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

template_df = pd.DataFrame(
    columns=[
        "Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp",
        "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%",
        "Lighting_%", "Machinery_%"
    ]
)

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>🌱 Smart AI Energy Dashboard for Industries</h1>",
    unsafe_allow_html=True
)

# ─────────────── Analysis Function ───────────────
def run_ai_energy_analysis(df):
    # Feature engineering & model
    df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]
    X = df[["Year", "Month_Num", "Temp_Delta", "Humidity",
            "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"]]
    y = df["Energy_kWh"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predictions & scores
    df["Predicted_Energy"] = model.predict(X)
    df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF
    df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

    # Benchmarks & top inefficiency
    peak_month = df.loc[df["Energy_kWh"].idxmax(), "Month"]
    monthly_avg = df["Energy_kWh"].mean()
    df["Benchmarking"] = df["Energy_kWh"].apply(
        lambda x: "Above Average" if x > monthly_avg else "Below Average"
    )

    ineff_areas = {
        "HVAC": (df["HVAC_%"] > 50).sum(),
        "Lighting": (df["Lighting_%"] > 50).sum(),
        "Machinery": (df["Machinery_%"] > 50).sum(),
    }
    top_ineff = max(ineff_areas, key=ineff_areas.get)

    # Smart recommendations
    def get_recommendation(row):
        recs = []
        base = row["Energy_kWh"]
        def savings(pct):
            red = pct * 0.20
            save_kwh = base * (red / 100)
            return save_kwh * TARIFF, save_kwh * CO2_PER_KWH

        if row["HVAC_%"] > 50:
            c, c2 = savings(row["HVAC_%"])
            recs.append(f"Optimize HVAC (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
        if row["Lighting_%"] > 50:
            c, c2 = savings(row["Lighting_%"])
            recs.append(f"Efficient lighting (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
        if row["Machinery_%"] > 50:
            c, c2 = savings(row["Machinery_%"])
            recs.append(f"Improve machinery (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
        if row["Efficiency_Score"] < 85:
            recs.append("Consider energy audits")

        return "; ".join(recs) if recs else "All systems efficient"

    df["Smart_Recommendation"] = df.apply(get_recommendation, axis=1)

    # Totals
    df["Est_Cost_Saved"] = df["Smart_Recommendation"]\
        .str.extractall(r"₹(\d+)")\
        .astype(float).sum(axis=1)
    df["Est_CO2_Saved"] = df["Smart_Recommendation"]\
        .str.extractall(r"([\d.]+) kg")\
        .astype(float).sum(axis=1)

    total_cost_saved = df["Est_Cost_Saved"].sum()
    total_co2_saved = df["Est_CO2_Saved"].sum()

    # In‑function display of per‑month recommendations
    st.markdown("### 🧠 AI Recommendations")
    for _, r in df.iterrows():
        st.markdown(
            f"""<div style='border:1px solid #ddd; padding:8px; margin:4px 0; border-radius:6px; background:#f9f9f9;'>
                <b>{r['Month']}</b>: {r['Smart_Recommendation']}<br>
                <i>Efficiency Score: {r['Efficiency_Score']}%</i>
            </div>""",
            unsafe_allow_html=True
        )

    return df, monthly_avg, peak_month, top_ineff, total_cost_saved, total_co2_saved

# ─────────────── Upload Section ───────────────
st.markdown("## 📁 Data Input Options")
with st.expander("📄 Upload CSV File"):
    csv_buf = io.StringIO()
    template_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        data=csv_buf.getvalue(),
        file_name="energy_input_template.csv",
        mime="text/csv"
    )
    uploaded = st.file_uploader("Upload your completed CSV file", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)

# ─────────────── Manual Entry ───────────────
st.markdown("### OR")
st.markdown("## ✍️ Manual Entry")

years = list(range(2000, 2031))
sel_year = st.selectbox("Select Year", years)
input_rows = []

for i, mon in enumerate(months):
    with st.expander(f"Month: {mon}"):
        e = st.number_input(f"Energy (kWh) - {mon}", value=0.0, key=f"e{i}")
        t_o = st.number_input(f"Outdoor Temp (°C) - {mon}", value=30.0, key=f"o{i}")
        t_i = st.number_input(f"Indoor Temp (°C) - {mon}", value=24.0, key=f"i{i}")
        h = st.number_input(f"Humidity (%) - {mon}", value=50.0, key=f"h{i}")
        oc = st.slider(f"Occupancy (%) - {mon}", 0, 100, 75, key=f"oc{i}")
        hv = st.slider(f"HVAC Usage (%) - {mon}", 0, 100, 40, key=f"hv{i}")
        li = st.slider(f"Lighting Usage (%) - {mon}", 0, 100, 30, key=f"li{i}")
        ma = st.slider(f"Machinery Usage (%) - {mon}", 0, 100, 30, key=f"ma{i}")

        input_rows.append([
            sel_year, mon, i+1, e, t_o, t_i, h, oc, hv, li, ma
        ])

if input_rows:
    df = pd.DataFrame(
        input_rows,
        columns=[
            "Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp",
            "Indoor_Temp", "Humidity", "Occupancy_%", "HVAC_%", "Lighting_%", "Machinery_%"
        ]
    )

# ─────────────── Run Analysis Buttons ───────────────
if st.button("💡 Run AI Analysis on Uploaded Data") and 'df' in locals():
    st.session_state.df_analyzed, \
    st.session_state.avg, \
    st.session_state.peak, \
    st.session_state.top_ineff, \
    st.session_state.cost_saved, \
    st.session_state.co2_saved = run_ai_energy_analysis(df)
    st.success("✅ AI analysis completed on uploaded data.")

if st.button("💡 Run AI Analysis on Manual Entry") and 'df' in locals():
    st.session_state.df_analyzed, \
    st.session_state.avg, \
    st.session_state.peak, \
    st.session_state.top_ineff, \
    st.session_state.cost_saved, \
    st.session_state.co2_saved = run_ai_energy_analysis(df)
    st.success("✅ AI analysis completed on manual input.")

# ─────────────── Persisted Visualization, KPIs & Report ───────────────
if "df_analyzed" in st.session_state:
    df2 = st.session_state.df_analyzed

    st.subheader("📊 Visualization")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔋 Energy (Actual vs Predicted)")
        df2_plot = df2.set_index(df2["Year"].astype(str) + "-" + df2["Month"])
        st.line_chart(df2_plot[["Energy_kWh", "Predicted_Energy"]])
    with c2:
        st.markdown("#### 💰 Cost (Actual vs Predicted)")
        st.line_chart(df2_plot[["Predicted_Cost"]])

    st.markdown("### ⚙️ Key Performance Indicators (KPIs)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Monthly Energy", f"{df2['Energy_kWh'].mean():.2f} kWh")
    k2.metric("Avg Monthly Cost", f"₹{df2['Predicted_Cost'].mean():.2f}")
    k3.metric("Total CO₂ Emitted", f"{df2['Energy_kWh'].sum() * CO2_PER_KWH:.2f} kg")
    k4.metric("Peak Load Month", st.session_state.peak)

    st.markdown("#### ⚡️ Monthly Efficiency Score")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = df2["Efficiency_Score"].apply(
        lambda x: "green" if x > 100 else ("orange" if x >= 90 else "red")
    )
    ax.bar(df2["Month"], df2["Efficiency_Score"], color=colors)
    ax.axhline(100, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Efficiency Score (%)")
    ax.set_title("Monthly Energy Efficiency")
    st.pyplot(fig)

    st.subheader("AI Summary Report")
    st.markdown(f"**Peak Load Month:** {st.session_state.peak}")
    st.markdown(f"**Average Energy Usage:** {st.session_state.avg:.2f} kWh")
    st.markdown(f"**Average Monthly Cost:** ₹{df2['Predicted_Cost'].mean():.2f}")
    st.markdown(f"**Total CO₂ Emissions:** {df2['Energy_kWh'].sum() * CO2_PER_KWH:.2f} kg")
    st.markdown(f"**Highest Efficiency Score:** {df2['Efficiency_Score'].max():.2f}")

    low = df2[df2["Efficiency_Score"] < 90]["Month"].tolist()
    if low:
        st.markdown(f"**Months with Low Efficiency (<90%):** {', '.join(low)}")
    else:
        st.markdown("**All months have good energy efficiency.**")

    # Potential Savings
    ineff = df2[df2["Efficiency_Score"] < 90].copy()
    ineff["Energy_Saved"] = ineff["Energy_kWh"] - ineff["Predicted_Energy"]
    ineff["Cost_Saved"] = ineff["Energy_Saved"] * TARIFF
    ineff["CO2_Saved"] = ineff["Energy_Saved"] * CO2_PER_KWH

    st.markdown("### Potential Savings if Low‑Efficiency Months Hit 100%")
    st.markdown(f"- **Annual Cost Savings:** ₹{ineff['Cost_Saved'].sum():.0f}")
    st.markdown(f"- **Annual CO₂ Reduction:** {ineff['CO2_Saved'].sum():.1f} kg")
    st.markdown(f"- **Top Inefficiency Area:** {st.session_state.top_ineff}")

    st.download_button(
        "Download Report as CSV",
        data=df2.to_csv(index=False),
        file_name="energy_ai_analysis.csv",
        mime="text/csv"
    )
