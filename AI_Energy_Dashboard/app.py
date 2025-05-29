import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
st.title("🌱 AI Energy Dashboard for Small Scale Industries")

# ─────────────── Session State Initialization ───────────────
if 'num_years' not in st.session_state:
    st.session_state.num_years = None
if 'selected_years' not in st.session_state:
    st.session_state.selected_years = []
if 'data_mode' not in st.session_state:
    st.session_state.data_mode = None

# ─────────────── AI Analysis Function ───────────────
# ─────────────── AI Analysis Function ───────────────
def run_ai_energy_analysis(df):
    df = df.copy()

    # Convert month names (if any) to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12,
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9,
        'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    if df['Month'].dtype == object:
        df['Month'] = df['Month'].map(lambda x: month_map.get(str(x).strip(), x)).astype(int)

    # Basic Calculations
    df['Energy_kWh'] = df['Electricity_Usage_Watts']
    df['Cost_INR'] = df['Energy_kWh'] * TARIFF
    df['CO2_kg'] = df['Energy_kWh'] * CO2_PER_KWH
    df['Temp_Delta'] = df['Avg_Internal_Temp_C'] - df['Avg_External_Temp_C']

    # Trends & Peak
    df['Trend_3M_kWh'] = df['Energy_kWh'].rolling(3, min_periods=1).mean()
    peak_idx = df['Energy_kWh'].idxmax()
    peak_year, peak_month, peak_val = df.loc[peak_idx, ['Year', 'Month', 'Energy_kWh']]

    # LSTM Cost Forecast
    df['Month_Index'] = (df['Year'] - df['Year'].min()) * 12 + df['Month']
    df = df.sort_values('Month_Index')
    cost_series = df['Cost_INR'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    cost_scaled = scaler.fit_transform(cost_series)

    X, y = [], []
    for i in range(len(cost_scaled) - 3):
        X.append(cost_scaled[i:i + 3])
        y.append(cost_scaled[i + 3])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(3, 1)),
        LSTM(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=80, verbose=0)

    last_seq = cost_scaled[-3:].reshape(1, 3, 1)
    next_cost_scaled = model.predict(last_seq)[0][0]
    next_cost = scaler.inverse_transform([[next_cost_scaled]])[0][0]

    # Recommendations
    avg_usage = df[['Machinery_Usage_Percent', 'Lighting_Usage_Percent', 'HVAC_Usage_Percent']].mean()
    top_comp = avg_usage.idxmax().replace('_Usage_Percent', '')
    recs = [
        f"🔺 Peak usage: {peak_val:.1f} kWh in {int(peak_year)}-{'%02d' % peak_month}",
        f"💡 Optimize {top_comp} usage (avg {avg_usage[top_comp + '_Usage_Percent']:.1f}%)",
        "💡 Adjust HVAC setpoints by 1-2°C",
        "💡 Implement LED lighting and sensor controls",
        "🌱 Conduct regular equipment maintenance"
    ]

    # Benchmark
    thresh = df['Energy_kWh'].mean() * 0.9
    df['Benchmark'] = df['Energy_kWh'].apply(lambda x: 'Good' if x <= thresh else 'High')

    # Plot 1: Energy
    st.subheader("📊 Energy Consumption & Trends")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    df['Label'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    ax1.plot(df['Label'], df['Energy_kWh'], marker='o', label='Actual kWh')
    ax1.plot(df['Label'], df['Trend_3M_kWh'], linestyle='--', label='3M Avg')
    ax1.set_xticklabels(df['Label'], rotation=45)
    ax1.set_ylabel('kWh')
    ax1.legend()
    st.pyplot(fig1)

    # Plot 2: Cost Trend
    st.subheader("📈 Cost Trend Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df['Label'], df['Cost_INR'], marker='x', color='orange')
    ax2.set_xticklabels(df['Label'], rotation=45)
    ax2.set_ylabel('Cost (INR)')
    st.pyplot(fig2)

    # KPIs
    st.subheader("💰 Cost Forecast & 🌎 CO₂ Emissions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Next Month Cost", f"₹{next_cost:.2f}")
    c2.metric("Total Cost", f"₹{df['Cost_INR'].sum():.2f}")
    c3.metric("Total CO₂", f"{df['CO2_kg'].sum():.2f} kg")

    st.subheader("⚙️ Key Performance Indicators")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Energy", f"{df['Energy_kWh'].sum():.1f} kWh")
    k2.metric("Avg Monthly kWh", f"{df['Energy_kWh'].mean():.1f}")
    k3.metric("Peak Month", f"{int(peak_year)}-{'%02d' % peak_month}")
    k4.metric("Efficiency Score", f"{(df['Trend_3M_kWh'].mean() / df['Energy_kWh'].mean() * 100):.1f}%")
    k5.metric("Benchmark", df['Benchmark'].iloc[-1])

    st.subheader("🔧 Recommendations to Optimize Energy")
    for r in recs:
        st.markdown(f"- {r}")

    # Download CSV
    csv_data = df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, "energy_report.csv", "text/csv")

    # Download PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = [Paragraph("AI Energy Analysis Report", styles['Title']), Spacer(1, 12)]
    elems.append(Paragraph(f"Total Energy: {df['Energy_kWh'].sum():.1f} kWh", styles['Normal']))
    elems.append(Paragraph(f"Total Cost: ₹{df['Cost_INR'].sum():.2f}", styles['Normal']))
    elems.append(Paragraph(f"Total CO₂: {df['CO2_kg'].sum():.2f} kg", styles['Normal']))
    elems.append(Spacer(1, 12))
    data = [df.columns.tolist()] + df.head(10).values.tolist()
    elems.append(Table(data))
    doc.build(elems)
    buffer.seek(0)
    st.download_button("Download PDF", buffer, "energy_report.pdf", "application/pdf")

    return df


# ─────────────── Step 1: Years Selection ───────────────
st.subheader("How many years of data would you like to analyze?")
years = list(range(1, 11))
num = st.selectbox("Please select number of years:", years, index=0)
st.session_state.num_years = num

# Step 1b: Select the exact years from 1990 to 2024 with multiselect limited by number chosen
available_years = list(range(1990, 2025))
selected_years = st.multiselect(
    f"Select exactly {num} year(s):",
    available_years,
    default=available_years[-num:]
)
# Validate selection length
if len(selected_years) != num:
    st.warning(f"Please select exactly {num} year(s).")
    st.stop()
else:
    st.session_state.selected_years = selected_years

# ─────────────── Step 2: Data Input Method ───────────────
st.subheader("Choose data input method:")
mode = st.radio("Select input type:", ["Upload CSV File", "Manual Entry"])
st.session_state.data_mode = mode

# ─────────────── Step 3A: CSV Upload ───────────────
if mode == "Upload CSV File":
    st.subheader("📄 Upload CSV File")
    template = pd.DataFrame(columns=[
        'Year', 'Month', 'Electricity_Usage_Watts', 'Avg_External_Temp_C',
        'Avg_Internal_Temp_C', 'Machinery_Usage_Percent',
        'Lighting_Usage_Percent', 'HVAC_Usage_Percent'
    ])

    # Fill template with blank rows for the selected years & 12 months each
    rows = []
    for y in selected_years:
        for m in range(1, 13):
            rows.append({
                'Year': y,
                'Month': m,
                'Electricity_Usage_Watts': np.nan,
                'Avg_External_Temp_C': np.nan,
                'Avg_Internal_Temp_C': np.nan,
                'Machinery_Usage_Percent': np.nan,
                'Lighting_Usage_Percent': np.nan,
                'HVAC_Usage_Percent': np.nan
            })
    template = pd.DataFrame(rows)

    buf = io.StringIO()
    template.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV Template", buf.getvalue(), "template.csv", "text/csv")

    upload = st.file_uploader("Upload your completed CSV", type="csv")
    if upload is not None:
        df_csv = pd.read_csv(upload)
        # Check if uploaded years and months match selection
        years_in_csv = sorted(df_csv['Year'].unique())
        if sorted(selected_years) != years_in_csv:
            st.error(f"Uploaded CSV years {years_in_csv} do not match your selected years {selected_years}.")
        elif not all(month in df_csv['Month'].values for month in range(1, 13)):
            st.error("Uploaded CSV must contain all 12 months for each selected year.")
        else:
            st.write(df_csv.head())
            if st.button("Run AI Analysis on CSV"):
                st.session_state.results = run_ai_energy_analysis(df_csv)

# ─────────────── Step 3B: Manual Entry ───────────────
elif mode == "Manual Entry":
    st.subheader("✍️ Manual Data Entry")
    entries = []
    for year in selected_years:
        st.markdown(f"#### Data for Year {year}")
        for m in range(1, 13):
            with st.expander(f"Month {m}"):
                watts = st.number_input(f"Electricity Usage (W) - {year}-{m}", min_value=0.0, key=f"w_{year}_{m}")
                ext = st.number_input(f"Outdoor Temp (°C) - {year}-{m}", value=30.0, key=f"ext_{year}_{m}")
                intl = st.number_input(f"Indoor Temp (°C) - {year}-{m}", value=24.0, key=f"int_{year}_{m}")
                mach = st.slider(f"Machinery_Usage_Percent - {year}-{m}", 0, 100, 30, key=f"mach_{year}_{m}")
                light = st.slider(f"Lighting_Usage_Percent - {year}-{m}", 0, 100, 30, key=f"light_{year}_{m}")
                hvac = st.slider(f"HVAC_Usage_Percent- {year}-{m}", 0, 100, 40, key=f"hvac_{year}_{m}")
                entries.append({
                    'Year': year,
                    'Month': m,
                    'Electricity_Usage_Watts': watts,
                    'Avg_External_Temp_C': ext,
                    'Avg_Internal_Temp_C': intl,
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

