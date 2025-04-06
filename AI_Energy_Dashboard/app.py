import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AI-Powered Energy Dashboard", layout="wide")

st.title("⚡ AI Energy Management Dashboard")
st.markdown("This AI dashboard helps predict energy usage, estimate costs, and generate smart energy-saving tips.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your monthly energy CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Uploaded Data")
    st.dataframe(df)

    # Features
    try:
        X = df[["Month_Num", "Avg_Temp", "Humidity", "Occupancy_%"]]
        y = df["Energy_kWh"]
    except KeyError:
        st.error("❌ Required columns: ['Month_Num', 'Avg_Temp', 'Humidity', 'Occupancy_%', 'Energy_kWh']")
    else:
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.subheader("📈 Model Evaluation")
        st.write(f"**MAE**: {mae:.2f} kWh")
        st.write(f"**RMSE**: {rmse:.2f} kWh")
        st.write(f"**R² Score**: {r2:.2f}")

        # Plot prediction vs actual
        st.subheader("📉 Actual vs Predicted Energy Usage")
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="Actual", marker="o")
        ax.plot(y_pred, label="Predicted", marker="x")
        ax.set_xlabel("Test Samples")
        ax.set_ylabel("Energy (kWh)")
        ax.legend()
        st.pyplot(fig)

        # AI Insights
        st.subheader("💡 AI Energy-Saving Suggestions")

        importances = model.feature_importances_
        features = X.columns
        sorted_indices = importances.argsort()[::-1]

        for i in sorted_indices:
            if features[i] == "Occupancy_%":
                st.write("- Optimize high occupancy periods with energy zoning.")
            elif features[i] == "Avg_Temp":
                st.write("- Control HVAC more efficiently during hotter months.")
            elif features[i] == "Humidity":
                st.write("- Use dehumidifiers or ventilation to reduce excess humidity load.")
            elif features[i] == "Month_Num":
                st.write("- Adjust energy settings seasonally to avoid peak load.")

        # Forecast next 3 months
        st.subheader("🔮 Future Forecast (Next 3 Months)")

        future_data = pd.DataFrame({
            "Month_Num": [13, 14, 15],
            "Avg_Temp": [32, 34, 35],
            "Humidity": [55, 60, 58],
            "Occupancy_%": [85, 90, 88]
        })

        future_prediction = model.predict(future_data)

        future_df = future_data.copy()
        future_df["Predicted_Energy_kWh"] = future_prediction
        st.dataframe(future_df)

        st.success("✅ AI model executed and optimized successfully!")
