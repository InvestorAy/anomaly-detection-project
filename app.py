import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# Load trained model + assets
# ----------------------------
model = load_model("autoencoder.keras")
scaler = joblib.load("scaler.pkl")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ----------------------------
# App UI
# ----------------------------
st.title("Network Anomaly Detection System")
st.write("Upload network traffic data to detect anomalies")

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(data.head())

    # ----------------------------
    # Preprocessing (must match training)
    # ----------------------------

    # Convert categorical to numeric
    data = pd.get_dummies(data)

    # Align columns with training data (VERY IMPORTANT FIX)
    data = data.reindex(columns=feature_columns, fill_value=0)

    # Scale data using saved scaler
    data_scaled = scaler.transform(data)

    # ----------------------------
    # Prediction
    # ----------------------------
    reconstructions = model.predict(data_scaled)

    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

    # Threshold (95th percentile)
    threshold = np.percentile(mse, 95)

    # Detect anomalies
    anomalies = mse > threshold

    # ----------------------------
    # Results
    # ----------------------------
    st.subheader("Results Summary")

    st.write("Threshold Value:", threshold)
    st.write("Total Records:", len(mse))
    st.write("Anomalies Detected:", np.sum(anomalies))

    # Show results table
    result_df = pd.DataFrame({
        "Reconstruction_Error": mse,
        "Anomaly": anomalies
    })

    st.write(result_df)

    # ----------------------------
    # Visualization
    # ----------------------------
    st.subheader("Anomaly Distribution")

    st.bar_chart(result_df["Anomaly"].value_counts())
