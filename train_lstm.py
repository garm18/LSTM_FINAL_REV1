import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import pymysql
import os
from datetime import datetime

ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][0])
    return np.array(X), np.array(y)

def inverse_transform(data, scaler):
    return scaler.inverse_transform(np.hstack((data.reshape(-1, 1), np.zeros((len(data), 2)))))[:, 0]

def train_and_save_model():
    # DB config
    db_config = {
        "host": "auth-db497.hstgr.io",
        "user": "u731251063_pgn",
        "password": "SmartMeter3",
        "database": "u731251063_pgn"
    }

    conn = pymysql.connect(**db_config)
    query = "SELECT created_at AS timestamp, signal_strength AS rssi FROM logs ORDER BY created_at ASC"
    df = pd.read_sql(query, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['rssi_smooth'] = df['rssi'].rolling(window=3, min_periods=1).mean()
    df['time_seconds'] = (df.index - df.index[0]).total_seconds()
    df['rssi_prev'] = df['rssi_smooth'].shift(1).fillna(method='bfill')

    features = ['rssi_smooth', 'rssi_prev', 'time_seconds']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = create_dataset(scaled, window_size=10)

    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=10)], verbose=0)

    # Save model & scaler
    model.save(os.path.join(ARTIFACT_DIR, 'lstm_rssi_model.h5'))
    with open(os.path.join(ARTIFACT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save test data
    test_start = 10 + train_size + val_size
    test_end = test_start + len(X_test)
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'timestamps': df.index[test_start:test_end],
        'actual_rssi': df['rssi'].iloc[test_start:test_end].values,
        'window_size': 10
    }
    with open(os.path.join(ARTIFACT_DIR, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    # Save test results
    y_test_inv = inverse_transform(y_test, scaler)
    y_pred_inv = inverse_transform(model.predict(X_test), scaler)

    pd.DataFrame({
        'timestamp': test_data['timestamps'],
        'actual_rssi': y_test_inv,
        'predicted_rssi': y_pred_inv
    }).to_csv(os.path.join(ARTIFACT_DIR, 'rssi_test_result_new.csv'), index=False)
