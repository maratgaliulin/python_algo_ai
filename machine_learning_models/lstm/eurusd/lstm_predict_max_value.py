import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
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
    X_train_raw = X_train_raw.reindex(columns=columns_order)
    X_test_raw = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test_raw = X_test_raw.reindex(columns=columns_order)
    X_validation_raw = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation_raw = X_validation_raw.reindex(columns=columns_order)

    y_train_raw = df[['y_60min_max']]
    y_test_raw = test_df[['y_60min_max']]
    y_validation_raw = validation_df[['y_60min_max']]

    # Normalize data (LSTMs are sensitive to scale)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # scaler_x.fit(X_train_raw)
    # scaler_x.fit(X_test_raw)

    # scaler_y.fit(y_train_raw)
    # scaler_y.fit(y_test_raw)

    # with open(base_dir + '/lstm_regressor_scaler_x_max.pkl', 'wb') as file:
    #         pickle.dump(scaler_x, file)

    # with open(base_dir + '/lstm_regressor_scaler_y_max.pkl', 'wb') as file:
    #         pickle.dump(scaler_y, file)

    # return

    scaled_data_train_x = scaler_x.fit_transform(X_train_raw)
    scaled_data_train_y = scaler_y.fit_transform(y_train_raw)

    scaled_data_test_x = scaler_x.fit_transform(X_test_raw)
    scaled_data_test_y = scaler_y.fit_transform(y_test_raw)

    scaled_data_validation_x = scaler_x.fit_transform(X_validation_raw)
    scaled_data_validation_y = scaler_y.fit_transform(y_validation_raw)
    
    SEQ_LENGTH = 12  # Time window (adjust based on your data)
    
    X_train, y_train = create_sequences(scaled_data_train_x, scaled_data_train_y, SEQ_LENGTH)    
    
    X_test, y_test = create_sequences(scaled_data_test_x, scaled_data_test_y, SEQ_LENGTH)
    
    X_validation, y_validation = create_sequences(scaled_data_validation_x, scaled_data_validation_y, SEQ_LENGTH)
    
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    validation_data = TensorDataset(X_validation, y_validation)

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    
    print('Deleting raw dataframes')
    del(X_train_raw)
    del(X_test_raw)
    del(X_validation_raw)
    print('Raw dataframes deleted')
    
    # print('shape of x_train:', X_train.shape)
    
    # return
    
    device = torch.device('cpu')
    
    complete_df_dir = base_dir + '/lstm_regressor_predict_max_values-2025-04-07.pkl'
    
    complete_files_exist = os.path.isfile(complete_df_dir)
    
    if(complete_files_exist):        
        with open(complete_df_dir, 'rb') as file:
            model = pickle.load(file)
    
    else:
        model = LSTMModel(
            input_size=len(columns_order),
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        epochs = 10
        batch_size = 32

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
                test_preds = []
                test_true = []
                for X_batch, y_batch in test_loader:
                    y_test_pred = model(X_batch)
                    test_preds.append(y_test_pred)
                    test_true.append(y_batch)
                
                test_preds = torch.cat(test_preds)
                test_true = torch.cat(test_true)
                test_loss = criterion(test_preds, test_true).item()
                test_losses.append(test_loss)

            model.eval()
            with torch.no_grad():
                validation_preds = []
                validation_true = []
                for X_batch, y_batch in validation_loader:
                    y_validation_pred = model(X_batch)
                    validation_preds.append(y_validation_pred)
                    validation_true.append(y_batch)
                
                validation_preds = torch.cat(validation_preds)
                validation_true = torch.cat(validation_true)
                validation_loss = criterion(validation_preds, validation_true).item()
                validation_losses.append(validation_loss)
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Validation loss: {validation_loss:.6f}')
                
        
        print('*******************************')
        print('After training on test sample:')
        print('*******************************')

        for epoch in range(epochs):
            # Training
            model.train()
            batch_losses = []
            for X_batch, y_batch in test_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            test_loss = np.mean(batch_losses)
            test_losses.append(test_loss)
            
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

            model.eval()
            with torch.no_grad():
                validation_preds = []
                validation_true = []
                for X_batch, y_batch in validation_loader:
                    y_validation_pred = model(X_batch)
                    validation_preds.append(y_validation_pred)
                    validation_true.append(y_batch)
                
                validation_preds = torch.cat(validation_preds)
                validation_true = torch.cat(validation_true)
                validation_loss = criterion(validation_preds, validation_true).item()
                validation_losses.append(validation_loss)
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Validation loss: {validation_loss:.6f}')

        
        
        with open(base_dir + '/lstm_regressor_predict_max_values-2025-04-07.pkl', 'wb') as file:
            pickle.dump(model, file)
    
    # return
    model.eval()
    
    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)
        y_validation_pred = model(X_validation)

    # Inverse transform predictions
    y_train_pred = scaler_y.inverse_transform(y_train_pred.numpy())
    y_test_pred = scaler_y.inverse_transform(y_test_pred.numpy())
    y_validation_predict = scaler_y.inverse_transform(y_validation_pred.numpy())

    y_train_actual = scaler_y.inverse_transform(y_train.numpy())
    y_test_actual = scaler_y.inverse_transform(y_test.numpy())
    y_validation_actual = scaler_y.inverse_transform(y_validation.numpy())
    
    # Calculate RMSE
    train_rmse = np.sqrt(np.mean((y_train_actual - y_train_pred)**2))
    test_rmse = np.sqrt(np.mean((y_test_actual - y_test_pred)**2))
    validation_rmse = np.sqrt(np.mean((y_validation_actual - y_validation_predict)**2))
    print(f'\nFinal Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f} | Validation RMSE: {validation_rmse:.2f}')

    
    # # --- Step 6: Plot Results ---
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_actual, label='Actual Max High (60min)', alpha=0.7)
    # plt.plot(y_test_pred, label='Predicted Max High', linestyle='--')
    # plt.title('PyTorch LSTM: Actual vs Predicted 60-Minute Maximum High Price')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()

    return