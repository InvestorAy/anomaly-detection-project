import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model

# Load model + scaler + columns
model = load_model("autoencoder.keras")
scaler = joblib.load("scaler.pkl")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

st.title("Anomaly Detection System")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    # Load data
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(data.head())

    # Encode categorical data
    data = pd.get_dummies(data)

    # ALIGN COLUMNS (FIX FOR YOUR ERROR)
    data = data.reindex(columns=feature_columns, fill_value=0)

    # Scale data (IMPORTANT: use saved scaler)
    data_scaled = scaler.transform(data)

    # Predict
    reconstructions = model.predict(data_scaled)

    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

    threshold = np.percentile(mse, 95)

    anomalies = mse > threshold

    st.subheader("Results")
    st.write("Threshold:", threshold)
    st.write("Anomaly Count:", np.sum(anomalies))

    st.subheader("Anomaly Flags")
    st.write(anomalies)
