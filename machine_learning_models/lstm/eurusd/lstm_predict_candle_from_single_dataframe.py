import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from machine_learning_models.classes.stock_predictor import StockPredictor
from machine_learning_models.lstm.lstm_model_class import LSTMModel

# Create Sequences for LSTM ---

def create_sequences(data_x, data_y, seq_length):

    X, y = [], []
    for i in range(len(data_x) - seq_length):
        X.append(data_x[i:i+seq_length])  # Input sequence (e.g., past 60 days)
        y.append(data_y[i+seq_length])    # Target (next day's price)
    # print('////////////////////')
    # print(y)
    # print('////////////////////')
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).view(-1, 1)

def predict_candle(df:pd.DataFrame, base_dir:str, column_for_y:str):

    columns_order = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "+DI",
        "-DI",
        "ADX",
        "ADL",
        "ATR_14",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "open_audusd",
        "high_audusd",
        "low_audusd",
        "close_audusd",
        "volume_audusd",
        "open_brentusd",
        "high_brentusd",
        "low_brentusd",
        "close_brentusd",
        "volume_brentusd",
        "open_cadusd",
        "high_cadusd",
        "low_cadusd",
        "close_cadusd",
        "volume_cadusd",
        "open_jpyusd",
        "high_jpyusd",
        "low_jpyusd",
        "close_jpyusd",
        "volume_jpyusd"
    ]

    columns_to_drop = [
        "open_plus_5min",
        "high_plus_5min",
        "low_plus_5min",
        "close_plus_5min",
        "open_plus_10min",    
        "high_plus_10min",
        "low_plus_10min",
        "close_plus_10min",
        "open_plus_15min",
        "high_plus_15min",
        "low_plus_15min",
        "close_plus_15min",
        "open_plus_20min",
        "high_plus_20min",
        "low_plus_20min",
        "close_plus_20min",
        "open_plus_25min",
        "high_plus_25min",
        "low_plus_25min",
        "close_plus_25min",
        "open_plus_30min",
        "high_plus_30min",
        "low_plus_30min",
        "close_plus_30min",
        "open_plus_35min",
        "high_plus_35min",
        "low_plus_35min",
        "close_plus_35min",
        "open_plus_40min",
        "high_plus_40min",
        "low_plus_40min",
        "close_plus_40min"
    ]
    
    X_train_raw = df.drop(columns=columns_to_drop)
    X_train_raw = X_train_raw.reindex(columns=columns_order)

    y_train_raw = df[[column_for_y]]
    

    # Normalize data (LSTMs are sensitive to scale)
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()

    scaler_x.fit(X_train_raw)

    scaler_y.fit(y_train_raw)

    with open(base_dir + '/lstm_regressor_scaler_x.pkl', 'wb') as file:
            pickle.dump(scaler_x, file)

    with open(base_dir + f'/lstm_regressor_scaler_y_{column_for_y}.pkl', 'wb') as file:
            pickle.dump(scaler_y, file)

    # return

    scaled_data_train_x = scaler_x.fit_transform(X_train_raw)
    scaled_data_train_y = scaler_y.fit_transform(y_train_raw)
    
    SEQ_LENGTH = 60  # Time window (adjust based on your data)
    
    X_train, y_train = create_sequences(scaled_data_train_x, scaled_data_train_y, SEQ_LENGTH)    
    
    train_data = TensorDataset(X_train, y_train)

    batch_size = 120
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    
    print('Deleting raw dataframes')
    del(X_train_raw)
    print('Raw dataframes deleted')
    
    # print('shape of x_train:', X_train.shape)
    
    # return
    
    device = torch.device('cpu')
    
    complete_df_dir = base_dir + f'/lstm_regressor_predict_candle_{column_for_y}.pkl'

    model = LSTMModel(
        input_size=len(columns_order),
        hidden_size=64,
        num_layers=2,
        output_size=1
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 1000
    batch_size = 120

    train_losses = []
    test_losses = []
    validation_losses = []

    print('*******************************')
    print('Before training on test sample:')
    print('*******************************')
    
    for epoch in range(epochs):
        # Training
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            train_preds = []
            train_true = []
            for X_batch, y_batch in train_loader:
                y_train_pred = model(X_batch)
                train_preds.append(y_train_pred)
                train_true.append(y_batch)
            
            train_preds = torch.cat(train_preds)
            train_true = torch.cat(train_true)
            train_loss = criterion(train_preds, train_true).item()
            train_losses.append(train_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}')
        
        
    with open(complete_df_dir, 'wb') as file:
        pickle.dump(model, file)
    
    # return
    model.eval()
    
    with torch.no_grad():
        y_train_pred = model(X_train)

    # Inverse transform predictions
    y_train_pred = scaler_y.inverse_transform(y_train_pred.numpy())

    y_train_actual = scaler_y.inverse_transform(y_train.numpy())
    
    # Calculate RMSE
    train_rmse = np.sqrt(np.mean((y_train_actual - y_train_pred)**2))
    print(f'\nFinal Train RMSE: {train_rmse:.6f}')

    return