import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    columns_order = ['open', 'open_minus_5min', 'open_minus_10min', 'open_minus_15min',
       'open_minus_20min', 'open_minus_25min', 'open_minus_30min',
       'open_minus_35min', 'open_minus_40min', 'open_minus_45min',
       'open_minus_50min', 'open_minus_55min', 'open_minus_60min', 'close',
       'close_minus_5min', 'close_minus_10min', 'close_minus_15min',
       'close_minus_20min', 'close_minus_25min', 'close_minus_30min',
       'close_minus_35min', 'close_minus_40min', 'close_minus_45min',
       'close_minus_50min', 'close_minus_55min', 'close_minus_60min', 'high',
       'high_minus_5min', 'high_minus_10min', 'high_minus_15min',
       'high_minus_20min', 'high_minus_25min', 'high_minus_30min',
       'high_minus_35min', 'high_minus_40min', 'high_minus_45min',
       'high_minus_50min', 'high_minus_55min', 'high_minus_60min', 'low',
       'low_minus_5min', 'low_minus_10min', 'low_minus_15min',
       'low_minus_20min', 'low_minus_25min', 'low_minus_30min',
       'low_minus_35min', 'low_minus_40min', 'low_minus_45min',
       'low_minus_50min', 'low_minus_55min', 'low_minus_60min', 'volume',
       'volume_minus_5min', 'volume_minus_10min', 'volume_minus_15min',
       'volume_minus_20min', 'volume_minus_25min', 'volume_minus_30min',
       'volume_minus_35min', 'volume_minus_40min', 'volume_minus_45min',
       'volume_minus_50min', 'volume_minus_55min', 'volume_minus_60min',
       'open_normalized', 'close_normalized', 'high_normalized',
       'low_normalized', 'volume_normalized', 'open_log', 'close_log',
       'high_log', 'low_log', '+DI', '-DI', 'ADX']
    
    X_train_raw = df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_train_raw = X_train.reindex(columns=columns_order)
    X_test_raw = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test_raw = X_test_raw.reindex(columns=columns_order)
    X_validation_raw = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation_raw = X_validation_raw.reindex(columns=columns_order)

    y_train_raw = df['y_60min_min']
    y_test_raw = test_df['y_60min_min']
    y_validation_raw = validation_df['y_60min_min']

    # Normalize data (LSTMs are sensitive to scale)
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data_train = scaler.fit_transform(X=X_train_raw, y=y_train_raw)
    scaled_data_test = scaler.fit_transform(X=X_test_raw, y=y_test_raw)
    scaled_data_validation = scaler.fit_transform(X=X_validation_raw, y=y_validation_raw)

    SEQ_LENGTH = 30  # Time window (adjust based on your data)
    
    X_train, y_train = create_sequences(scaled_data_train, SEQ_LENGTH)
    y_train = y_train[:, :1]
    
    X_test, y_test = create_sequences(scaled_data_test, SEQ_LENGTH)
    y_test = y_test[:, :1]
    
    X_validation, y_validation = create_sequences(scaled_data_validation, SEQ_LENGTH)
    y_validation = y_validation[:, :1]
    
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
            
    
    print('*******************************')
    print('Before training on test sample:')
    print('*******************************')
    
    model.eval()
    
    train_predict = model(X_train)
    test_predict = model(X_test)
    validation_predict = model(X_validation)
    
    train_predict = scaler.inverse_transform(train_predict.detach().numpy())
    y_train_actual = scaler.inverse_transform(y_train.detach().numpy())
    
    test_predict = scaler.inverse_transform(test_predict.detach().numpy())
    y_test_actual = scaler.inverse_transform(y_test.detach().numpy())
    
    validation_predict = scaler.inverse_transform(validation_predict.detach().numpy())
    y_validation_actual = scaler.inverse_transform(y_validation.detach().numpy())

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_train_actual, label='Actual Train Prices')
    plt.plot(train_predict, label='Predicted Train Prices')    
    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)), y_test_actual, label='Actual Test Prices')    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)), test_predict, label='Predicted Test Prices')
    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)+len(y_validation_actual)), y_validation_actual, label='Actual Validation Prices')    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)+len(y_validation_actual)), validation_predict, label='Predicted Validation Prices')
    
    
    plt.legend()
    plt.show()
    
    model.train(mode=True)

    for epoch in range(epochs):
        for i in range(0, len({X_test}), batch_size):
            batch_X = {X_test}[i:i+batch_size].to(device)
            batch_y = y_test[i:i+batch_size].to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    
    print('*******************************')
    print('Before training on test sample:')
    print('*******************************')
    
    model.eval()
    
    train_predict = model(X_train)
    test_predict = model(X_test)
    validation_predict = model(X_validation)
    
    train_predict = scaler.inverse_transform(train_predict.detach().numpy())
    y_train_actual = scaler.inverse_transform(y_train.detach().numpy())
    
    test_predict = scaler.inverse_transform(test_predict.detach().numpy())
    y_test_actual = scaler.inverse_transform(y_test.detach().numpy())
    
    validation_predict = scaler.inverse_transform(validation_predict.detach().numpy())
    y_validation_actual = scaler.inverse_transform(y_validation.detach().numpy())

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_train_actual, label='Actual Train Prices')
    plt.plot(train_predict, label='Predicted Train Prices')    
    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)), y_test_actual, label='Actual Test Prices')    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)), test_predict, label='Predicted Test Prices')
    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)+len(y_validation_actual)), y_validation_actual, label='Actual Validation Prices')    
    plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)+len(y_validation_actual)), validation_predict, label='Predicted Validation Prices')
    
    
    plt.legend()
    plt.show()