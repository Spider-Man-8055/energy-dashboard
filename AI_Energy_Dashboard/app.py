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

# ─────────────── Streamlit Setup ───────────────
st.set_page_config(page_title="AI Energy Dashboard", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #1e1e1e; color: #f0f0f0; }
    .stButton>button { background-color: #2e8b57; color: white; }
    .stDownloadButton>button { background-color: #2e8b57; color: white; }
    </style>
    """, unsafe_allow_html=True
)
st.title("🌱 Smart AI Energy Dashboard for Industries")

# ─────────────── Session State Initialization ───────────────
if 'num_years' not in st.session_state:
    st.session_state.num_years = None
if 'data_mode' not in st.session_state:
    st.session_state.data_mode = None

# ─────────────── AI Analysis Function ───────────────
def run_ai_energy_analysis(df):
    # Preprocess
    df = df.copy()
    df['Energy_kWh'] = df['Electricity_Usage_Watts'] / 1000
    df['Cost_INR'] = df['Energy_kWh'] * TARIFF
    df['CO2_kg'] = df['Energy_kWh'] * CO2_PER_KWH
    df['Temp_Delta'] = df['Internal_Temp_C'] - df['External_Temp_C']

    # Features & target
    features = ['Year', 'Month', 'Electricity_Usage_Watts', 'Internal_Temp_C',
                'External_Temp_C', 'Machinery_Usage_Percent',
                'Lighting_Usage_Percent', 'HVAC_Usage_Percent']
    X = df[features].values
    y = df['Electricity_Usage_Watts'].values

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Build TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y, epochs=50, verbose=0)

    # Predictions
    df['Predicted_Usage_Watts'] = model.predict(X_scaled).flatten()
    df['Predicted_kWh'] = df['Predicted_Usage_Watts'] / 1000

    # Peak usage month
    peak_idx = df['Electricity_Usage_Watts'].idxmax()
    peak_info = df.loc[peak_idx, ['Year', 'Month', 'Electricity_Usage_Watts']]

    # Recommendations
    recs = []
    high_consumption = df['Electricity_Usage_Watts'].max()
    high_month = df.loc[df['Electricity_Usage_Watts'].idxmax(), 'Month']
    recs.append(f"🔺 Highest consumption: {high_consumption:.0f} W in Month {high_month}")
    recs.append("💡 Consider optimizing equipment usage during peak months.")
    recs.append("💡 Adjust HVAC setpoints by 1-2°C to save energy.")
    recs.append("💡 Use LED lighting and auto-shutoff sensors.")
    recs.append("🌱 Conduct regular maintenance to improve efficiency.")

    # Display Results
    st.subheader("📊 Usage Charts")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Month'].astype(str), df['Electricity_Usage_Watts'], label='Actual (W)')
    ax.plot(df['Month'].astype(str), df['Predicted_Usage_Watts'], label='Predicted (W)', linestyle='--')
    ax.set_xlabel("Month")
    ax.set_ylabel("Electricity Usage (W)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("⚙️ Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Energy (kWh)", f"{df['Energy_kWh'].sum():.2f}")
    col2.metric("Total Cost (INR)", f"₹{df['Cost_INR'].sum():.2f}")
    col3.metric("Total CO₂ (kg)", f"{df['CO2_kg'].sum():.2f}")

    st.subheader("🔧 Recommendations to Reduce Usage")
    for r in recs:
        st.markdown(f"- {r}")

    return df

# ─────────────── Step 1: Years Selection ───────────────
st.subheader("Step 1: How many years of data would you like to analyze?")
years = list(range(1, 6))
num = st.selectbox("Please select number of years:", years, index=0)
st.session_state.num_years = num

# ─────────────── Step 2: Data Input Method ───────────────
st.subheader("Step 2: Choose data input method:")
mode = st.radio("Select input type:", ["Upload CSV File", "Manual Entry"])
st.session_state.data_mode = mode

# ─────────────── Step 3A: CSV Upload ───────────────
if mode == "Upload CSV File":
    st.subheader("📄 Upload CSV File")
    template = pd.DataFrame(columns=[
        'Year', 'Month', 'Electricity_Usage_Watts', 'External_Temp_C',
        'Internal_Temp_C', 'Machinery_Usage_Percent',
        'Lighting_Usage_Percent', 'HVAC_Usage_Percent'
    ])
    buf = io.StringIO()
    template.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV Template", buf.getvalue(), "template.csv", "text/csv")

    upload = st.file_uploader("Upload your completed CSV", type="csv")
    if upload is not None:
        df_csv = pd.read_csv(upload)
        st.write(df_csv.head())
        if st.button("Run AI Analysis on CSV"):
            st.session_state.results = run_ai_energy_analysis(df_csv)

# ─────────────── Step 3B: Manual Entry ───────────────
elif mode == "Manual Entry":
    st.subheader("✍️ Manual Data Entry")
    entries = []
    years_list = []
    for i in range(st.session_state.num_years):
        year = st.number_input(f"Enter Year #{i+1}", min_value=1990, max_value=2100, value=2020)
        years_list.append(year)
        st.markdown(f"#### Data for Year {year}")
        for m in range(1, 13):
            with st.expander(f"Month {m}"):
                watts = st.number_input(f"Electricity Usage (W) - {year}-{m}", min_value=0.0, key=f"w_{year}_{m}")
                ext = st.number_input(f"Outdoor Temp (°C) - {year}-{m}", value=30.0, key=f"ext_{year}_{m}")
                intl = st.number_input(f"Indoor Temp (°C) - {year}-{m}", value=24.0, key=f"int_{year}_{m}")
                mach = st.slider(f"Machinery Usage (%) - {year}-{m}", 0, 100, 30, key=f"mach_{year}_{m}")
                light = st.slider(f"Lighting Usage (%) - {year}-{m}", 0, 100, 30, key=f"light_{year}_{m}")
                hvac = st.slider(f"HVAC Usage (%) - {year}-{m}", 0, 100, 40, key=f"hvac_{year}_{m}")
                entries.append({
                    'Year': year,
                    'Month': m,
                    'Electricity_Usage_Watts': watts,
                    'External_Temp_C': ext,
                    'Internal_Temp_C': intl,
                    'Machinery_Usage_Percent': mach,
                    'Lighting_Usage_Percent': light,
                    'HVAC_Usage_Percent': hvac
                })
    if st.button("Run AI Analysis on Manual Data"):
        df_manual = pd.DataFrame(entries)
        st.session_state.results = run_ai_energy_analysis(df_manual)

# ─────────────── Results Section ───────────────
if 'results' in st.session_state:
    st.subheader("✅ AI Analysis Results")
    st.write(st.session_state.results.head())
