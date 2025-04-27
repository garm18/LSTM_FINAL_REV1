# file: forecast_rssi.py

import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# File paths
MODEL_PATH = "artifacts/lstm_rssi_model.h5"
SCALER_PATH = "artifacts/scaler.pkl"
TEST_DATA_PATH = "artifacts/test_data.pkl"

# Load model and scaler
def load_model_and_scaler():
    model = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Forecasting function
def forecast_next_minutes(minutes=30):
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)

    X_last = test_data['X_test'][-1]  # shape (window_size, n_features)
    last_timestamp = pd.to_datetime(test_data['timestamps'][-1])
    window_size = test_data['window_size']

    model, scaler = load_model_and_scaler()

    predictions = []
    timestamps = []
    current_input = X_last.copy()
    step_seconds = 60  # 1 minute

    for i in range(minutes):
        pred_scaled = model.predict(np.array([current_input]), verbose=0)

        # Inverse transform
        pred_full = np.hstack((pred_scaled, current_input[-1, 1:].reshape(1, -1)))
        pred_rssi = scaler.inverse_transform(np.hstack((pred_scaled, [[0, 0]])))[0, 0]

        predictions.append(round(float(pred_rssi), 2))
        timestamps.append((last_timestamp + timedelta(minutes=i+1)).strftime('%Y-%m-%d %H:%M:%S'))

        # Update input: shift window and append new predicted row
        time_next = current_input[-1, 2] + step_seconds
        new_input = np.array([pred_scaled[0, 0], current_input[-1, 0], time_next])  # [rssi_smooth, rssi_prev, time_seconds]
        new_input_scaled = scaler.transform(new_input.reshape(1, -1))

        current_input = np.vstack([current_input[1:], new_input_scaled])

    return pd.DataFrame({
        'timestamp': timestamps,
        'forecasted_rssi': predictions
    })
