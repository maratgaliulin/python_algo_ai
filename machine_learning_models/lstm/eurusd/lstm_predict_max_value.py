import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

# Create Sequences for LSTM ---

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # Input sequence (e.g., past 60 days)
        y.append(data[i+seq_length])    # Target (next day's price)
    return np.array(X), np.array(y)

def predict_max_value_with_lstm_model(df:pd.DataFrame, test_df:pd.DataFrame, validation_df:pd.DataFrame, base_dir:str):

    X_train_raw = df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test_raw = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation_raw = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train_raw = df['y_60min_max']
    y_test_raw = test_df['y_60min_max']
    y_validation_raw = validation_df['y_60min_max']

    # Normalize data (LSTMs are sensitive to scale)
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data_train = scaler.fit_transform(X=X_train_raw, y=y_train_raw)
    scaled_data_test = scaler.fit_transform(X=X_test_raw, y=y_test_raw)
    scaled_data_validation = scaler.fit_transform(X=X_validation_raw, y=y_validation_raw)

    SEQ_LENGTH = 30  # Time window (adjust based on your data)
    X_train, y_train = create_sequences(scaled_data_train, SEQ_LENGTH)
    X_test, y_test = create_sequences(scaled_data_train, SEQ_LENGTH)
    X_validation, y_validation = create_sequences(scaled_data_train, SEQ_LENGTH)

    # X_train, X_test = X[:split_idx], X[split_idx:]
    # y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for LSTM input: [samples, timesteps, features]
    print('Before reshaping')
    print(X_train)
    print(y_train)


    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print('After reshaping')
    print(X_train)
    print(y_train)

    # --- Step 3: Build LSTM Model ---
    # model = Sequential([
    #     LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),  # First LSTM layer
    #     Dropout(0.2),  # Prevent overfitting
    #     LSTM(50, return_sequences=False),  # Second LSTM layer
    #     Dense(25),  # Dense layer
    #     Dense(1)   # Output layer (predicted price)
    # ])

    # model.compile(optimizer="adam", loss="mse")  # Mean Squared Error for regression

    # # --- Step 4: Train the Model ---
    # early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    # history = model.fit(
    #     X_train, y_train,
    #     epochs=100,
    #     batch_size=32,
    #     validation_data=(X_test, y_test),
    #     callbacks=[early_stop],
    #     verbose=1
    # )

    # # --- Step 5: Evaluate & Predict ---
    # train_loss = model.evaluate(X_train, y_train, verbose=0)
    # test_loss = model.evaluate(X_test, y_test, verbose=0)
    # print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # # Predict on test data
    # predictions = model.predict(X_test)
    # predictions = scaler.inverse_transform(predictions)  # Undo scaling
    # y_test_actual = scaler.inverse_transform(y_test)     # Actual values