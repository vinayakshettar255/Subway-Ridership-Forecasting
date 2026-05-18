import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1,1))

    X, y = [], []
    for i in range(3, len(data_scaled)):
        X.append(data_scaled[i-3:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=50, verbose=0)

    return model, scaler

def predict_future(model, scaler, data, steps=6):
    input_seq = data[-3:]
    predictions = []

    for _ in range(steps):
        scaled = scaler.transform(input_seq.reshape(-1,1))
        pred = model.predict(scaled.reshape(1,3,1), verbose=0)
        val = scaler.inverse_transform(pred)[0][0]

        predictions.append(val)
        input_seq = np.append(input_seq[1:], val)

    return predictions