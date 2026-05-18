from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from model import train_lstm, predict_future

app = Flask(__name__)
CORS(app)

df = pd.read_csv("data/ridership.csv")

df = pd.read_csv("data/ridership.csv")

@app.route("/stations")
def stations():
    return df.to_json(orient="records")

@app.route("/forecast")
def forecast():
    data = df["daily_avg"].values

    model, scaler = train_lstm(data)
    future = predict_future(model, scaler, data)

    return jsonify({
        "history": data.tolist(),
        "forecast": future
    })

if __name__ == "__main__":
    app.run(debug=True)