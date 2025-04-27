from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

ARTIFACT_DIR = 'artifacts'
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'lstm_rssi_model.h5')
SCALER_PATH = os.path.join(ARTIFACT_DIR, 'scaler.pkl')
TEST_DATA_PATH = os.path.join(ARTIFACT_DIR, 'test_data.pkl')

# 🔹 Health Check
@app.route('/health', methods=['GET'])
def health_check():
    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)
    return jsonify({
        'status': 'healthy' if model_exists and scaler_exists else 'unhealthy',
        'model_found': model_exists,
        'scaler_found': scaler_exists
    })

# 🔹 Train Model
@app.route('/train-lstm', methods=['GET'])
def train_lstm():
    try:
        import train_lstm as trainer
        trainer.train_and_save_model()
        return jsonify({'message': 'Model retrained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🔹 Predict Model
@app.route('/predict-rssi', methods=['GET'])
def predict_rssi():
    try:
        model = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(TEST_DATA_PATH, 'rb') as f:
            test_data = pickle.load(f)

        X_test = test_data['X_test']
        y_test = test_data['y_test']
        timestamps = test_data['timestamps']

        y_pred = model.predict(X_test)

        def inverse(data):
            return scaler.inverse_transform(np.hstack((data.reshape(-1, 1), np.zeros((len(data), 2)))))[:, 0]

        y_actual_inv = inverse(y_test)
        y_pred_inv = inverse(y_pred)

        results = []
        for ts, act, pred in zip(timestamps, y_actual_inv, y_pred_inv):
            results.append({
                'timestamp': pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S'),
                'actual_rssi': round(float(act), 2),
                'predicted_rssi': round(float(pred), 2)
            })

        return jsonify({'data': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🔹 Forecast RSSI (forward prediction)
@app.route('/forecast-rssi', methods=['GET'])
def forecast_rssi():
    try:
        from forecast_rssi import forecast_next_minutes

        minutes = int(request.args.get('minutes', 30))  # default = 30
        forecast_df = forecast_next_minutes(minutes=minutes)

        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_data.append({
                'timestamp': row['timestamp'],
                'forecasted_rssi': row['forecasted_rssi']
            })

        return jsonify({'forecast': forecast_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
