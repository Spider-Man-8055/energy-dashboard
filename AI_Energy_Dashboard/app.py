import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Function to load CSV file
@st.cache
def load_csv(file):
    data = pd.read_csv(file)
    data['Year'] = data['Year'].astype(int)  # Ensure 'Year' is in integer format
    return data

# Streamlit dashboard layout
st.title("AI-powered Energy Management Dashboard")

# File upload section
st.subheader("Upload your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Manual data entry section
st.subheader("Enter Data Manually")
year = st.number_input('Enter Year:', min_value=2000, max_value=2100, value=2023, step=1)
month = st.selectbox('Select Month:', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
energy_kWh = st.number_input('Enter Energy (kWh):')
cost_inr = st.number_input('Enter Cost (INR):')
avg_temp = st.number_input('Enter Average Temperature (°C):')
indoor_temp = st.number_input('Enter Indoor Temperature (°C):')
humidity = st.number_input('Enter Humidity (%):')
occupancy = st.number_input('Enter Occupancy (%):')
hvac = st.number_input('Enter HVAC Usage (%):')
lighting = st.number_input('Enter Lighting Usage (%):')
machinery = st.number_input('Enter Machinery Usage (%):')

# Create a DataFrame for manual entry
manual_data = {
    'Year': [year],
    'Month': [month],
    'Energy_kWh': [energy_kWh],
    'Cost_INR': [cost_inr],
    'Avg_Temp': [avg_temp],
    'Indoor_Temp': [indoor_temp],
    'Humidity': [humidity],
    'Occupancy_%': [occupancy],
    'HVAC_%': [hvac],
    'Lighting_%': [lighting],
    'Machinery_%': [machinery]
}

manual_df = pd.DataFrame(manual_data)

# Load and process the CSV if uploaded
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.write("CSV Data Preview", df.head())
else:
    df = manual_df

# Combine uploaded CSV data with manual data entry
data = pd.concat([df, manual_df], ignore_index=True)  # Combine both data sources

# Feature engineering: Include 'Year' and 'Month' for better predictions
data['Month_Num'] = data['Month'].apply(lambda x: pd.to_datetime(x, format='%b').month)

# Model Training (Random Forest Regressor)
X = data[['Year', 'Month_Num', 'Avg_Temp', 'Humidity', 'Occupancy_%', 'HVAC_%', 'Lighting_%', 'Machinery_%']]
y = data['Energy_kWh']  # Target variable

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
data['Predicted_Energy_kWh'] = predictions

# AI Recommendations (based on predicted data)
st.subheader("AI Recommendations for Energy Savings")
predicted_energy = data['Predicted_Energy_kWh'].mean()
actual_energy = data['Energy_kWh'].mean()

# Generate recommendations
if predicted_energy < actual_energy:
    recommendation = "Great! Your energy consumption is predicted to be lower than usual. Keep up the good work!"
else:
    recommendation = "Energy consumption is predicted to rise. Consider optimizing HVAC usage, lighting, or machinery usage."

st.write(f"Prediction-based Recommendation: {recommendation}")

# Visualizations
st.subheader("Energy Consumption Visualization")

# Plot Energy Consumption Over Multiple Years
plt.figure(figsize=(10, 6))
for year in data['Year'].unique():
    year_data = data[data['Year'] == year]
    plt.plot(year_data['Month_Num'], year_data['Energy_kWh'], label=f'Year {year}')
    
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Energy Consumption Over Multiple Years')
plt.legend()
st.pyplot(plt)

# Plot Predicted vs Actual Energy Consumption
plt.figure(figsize=(10, 6))
plt.plot(data['Month_Num'], data['Energy_kWh'], label='Actual Energy Consumption', color='b')
plt.plot(data['Month_Num'], data['Predicted_Energy_kWh'], label='Predicted Energy Consumption', color='r', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Predicted vs Actual Energy Consumption')
plt.legend()
st.pyplot(plt)

# Generate Future Predictions for Multiple Years
st.subheader("Future Predictions for Energy Consumption")
future_years = [2024, 2025]  # Example future years
future_months = list(range(1, 13))  # All months (1 to 12)

future_predictions = []
for year in future_years:
    for month in future_months:
        features = [year, month, avg_temp, humidity, occupancy, hvac, lighting, machinery]
        predicted_energy = model.predict([features])[0]
        future_predictions.append({'Year': year, 'Month': month, 'Predicted Energy (kWh)': predicted_energy})

# Convert future predictions to DataFrame
future_predictions_df = pd.DataFrame(future_predictions)
st.write(future_predictions_df)

# Optionally download prediction data
st.download_button('Download Future Predictions', future_predictions_df.to_csv(index=False), file_name='future_predictions.csv')

