import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io

# ─────────────── Constants ───────────────
TARIFF = 8.5  # INR per kWh
CO2_PER_KWH = 0.82
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ─────────────── Streamlit Setup ───────────────
st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;color:#2E8B57;'>🌱 Smart AI Energy Dashboard for Industries</h1>", unsafe_allow_html=True)

# ─────────────── AI Analysis Function ───────────────
def run_ai_energy_analysis(df):
    df = df.copy()
    df["Cost_INR"] = df["Energy_kWh"] * TARIFF
    df["Temp_Delta"] = df["Indoor_Temp"] - df["Avg_Temp"]

    features = ["Year", "Month_Num", "Temp_Delta", "HVAC_%", "Lighting_%", "Machinery_%"]
    target = "Energy_kWh"

    X = df[features].values
    y = df[target].values

    # Normalize input features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # TensorFlow Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y, epochs=100, verbose=0)

    df["Predicted_Energy"] = model.predict(X_scaled).flatten()
    df["Predicted_Cost"] = df["Predicted_Energy"] * TARIFF
    df["Efficiency_Score"] = np.round((df["Predicted_Energy"] / df["Energy_kWh"]) * 100, 2)

    peak_month = df.loc[df["Energy_kWh"].idxmax(), "Month"]
    monthly_avg = df["Energy_kWh"].mean()
    df["Benchmarking"] = df["Energy_kWh"].apply(lambda x: "Above Average" if x > monthly_avg else "Below Average")

    ineff_counts = {
        "HVAC": (df["HVAC_%"] > 50).sum(),
        "Lighting": (df["Lighting_%"] > 50).sum(),
        "Machinery": (df["Machinery_%"] > 50).sum()
    }
    top_ineff = max(ineff_counts, key=ineff_counts.get)

    def compute_row(row):
        recs, cost_sum, co2_sum = [], 0.0, 0.0
        base = row["Energy_kWh"]
        def savings(pct):  # 20% savings
            saved_kwh = base * (pct * 0.20) / 100
            return saved_kwh * TARIFF, saved_kwh * CO2_PER_KWH

        for area in ["HVAC_%", "Lighting_%", "Machinery_%"]:
            if row[area] > 50:
                c, c2 = savings(row[area])
                label = area.replace("_%", "")
                recs.append(f"Optimize {label} (Save ₹{c:.0f}, {c2:.1f} kg CO₂/mo)")
                cost_sum += c
                co2_sum += c2

        if row["Efficiency_Score"] < 85:
            recs.append("Consider energy audits")

        text = "; ".join(recs) if recs else "All systems efficient"
        return pd.Series([text, cost_sum, co2_sum], index=["Smart_Recommendation", "Est_Cost_Saved", "Est_CO2_Saved"])

    rec_df = df.apply(compute_row, axis=1)
    df = pd.concat([df, rec_df], axis=1)

    return df, monthly_avg, peak_month, top_ineff, df["Est_Cost_Saved"].sum(), df["Est_CO2_Saved"].sum()

# ─────────────── Input Section ───────────────
st.markdown("## 📁 Data Input Options")

template_df = pd.DataFrame(columns=[
    "Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp",
    "Indoor_Temp", "HVAC_%", "Lighting_%", "Machinery_%"
])

with st.expander("📄 Upload CSV File"):
    buf = io.StringIO()
    template_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV Template", data=buf.getvalue(), file_name="energy_input_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your completed CSV file", type="csv")
    if uploaded is not None:
        try:
            st.session_state.df_uploaded = pd.read_csv(uploaded)
            st.success("✅ CSV file uploaded successfully!")
            if st.button("💡 Run AI Analysis on Uploaded Data"):
                (st.session_state.df_analyzed,
                 st.session_state.avg,
                 st.session_state.peak,
                 st.session_state.top_ineff,
                 st.session_state.cost_saved,
                 st.session_state.co2_saved) = run_ai_energy_analysis(st.session_state.df_uploaded)
                st.success("✅ AI analysis completed on uploaded data.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ─────────────── Manual Input ───────────────
st.markdown("### OR")
st.markdown("## ✍️ Manual Entry")

years = list(range(2000, 2031))
sel_year = st.selectbox("Select Year", years)
rows = []

for i, m in enumerate(months):
    with st.expander(f"Month: {m}"):
        energy = st.number_input(f"Energy (kWh) - {m}", value=0.0, key=f"e_{i}")
        outdoor_temp = st.number_input(f"Outdoor Temp (°C) - {m}", value=30.0, key=f"at_{i}")
        indoor_temp = st.number_input(f"Indoor Temp (°C) - {m}", value=24.0, key=f"it_{i}")
        hvac = st.slider(f"HVAC Usage (%) - {m}", 0, 100, 40, key=f"hv_{i}")
        lighting = st.slider(f"Lighting Usage (%) - {m}", 0, 100, 30, key=f"li_{i}")
        machinery = st.slider(f"Machinery Usage (%) - {m}", 0, 100, 30, key=f"ma_{i}")
        rows.append([sel_year, m, i+1, energy, outdoor_temp, indoor_temp, hvac, lighting, machinery])

if rows and st.button("💡 Run AI Analysis on Manual Entry"):
    df_manual = pd.DataFrame(rows, columns=[
        "Year", "Month", "Month_Num", "Energy_kWh", "Avg_Temp",
        "Indoor_Temp", "HVAC_%", "Lighting_%", "Machinery_%"
    ])
    (st.session_state.df_analyzed,
     st.session_state.avg,
     st.session_state.peak,
     st.session_state.top_ineff,
     st.session_state.cost_saved,
     st.session_state.co2_saved) = run_ai_energy_analysis(df_manual)
    st.success("✅ AI analysis completed on manual input.")

# ─────────────── Display Results ───────────────
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
        st.line_chart(df2_plot[["Cost_INR", "Predicted_Cost"]])

    st.markdown("### ⚙️ Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Monthly Energy", f"{df2['Energy_kWh'].mean():.2f} kWh")
    k2.metric("Avg Monthly Cost", f"₹{df2['Predicted_Cost'].mean():.2f}")
    k3.metric("Total CO₂ Emitted", f"{(df2['Energy_kWh'] * CO2_PER_KWH).sum():.2f} kg")
    k4.metric("🔺 Peak Load Month", st.session_state.peak)

    st.markdown("#### ⚡️ Monthly Efficiency Score")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = df2["Efficiency_Score"].apply(lambda x: "green" if x > 100 else ("orange" if x >= 90 else "red"))
    ax.bar(df2["Month"], df2["Efficiency_Score"], color=colors)
    ax.axhline(100, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Efficiency Score (%)")
    ax.set_title("Monthly Energy Efficiency")
    st.pyplot(fig)

    st.subheader("AI Summary Report")
    st.markdown(f"**Peak Load Month:** {st.session_state.peak}")
    st.markdown(f"**Average Energy Usage:** {st.session_state.avg:.2f} kWh")
    st.markdown(f"**Average Monthly Cost:** ₹{df2['Predicted_Cost'].mean():.2f}")
    st.markdown(f"**Total CO₂ Emissions:** {(df2['Energy_kWh'] * CO2_PER_KWH).sum():.2f} kg")

 
