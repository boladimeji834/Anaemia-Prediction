# app.py (Flask version)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from data_preprocessing import preprocess_data

app = Flask(__name__)
model = joblib.load("models/anaemia_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])  # assuming data comes in as JSON
    X, _ = preprocess_data(df)
    prediction = model.predict(X)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
