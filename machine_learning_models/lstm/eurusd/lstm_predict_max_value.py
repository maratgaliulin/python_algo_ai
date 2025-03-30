import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from machine_learning_models.classes.stock_predictor import StockPredictor

# Create Sequences for LSTM ---

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # Input sequence (e.g., past 60 days)
        y.append(data[i+seq_length])    # Target (next day's price)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

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
    y_train = y_train[:, :1]
    X_test, y_test = create_sequences(scaled_data_test, SEQ_LENGTH)
    X_validation, y_validation = create_sequences(scaled_data_validation, SEQ_LENGTH)
    
    # print('shape of x_train:', X_train.shape)
    
    # return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StockPredictor(input_size=77).to(device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    batch_size = 32
    
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')