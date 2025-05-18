import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


# Примечания:

# 1. Как действовать в переходной зоной (переход от тренда к флэту) в скользящем окне

def create_sequences(data_x, seq_length):
    X = []
    for i in range(len(data_x) - seq_length):
        X.append(data_x[i:i+seq_length])  # Input sequence (e.g., past 60 days)
    # print(X)
    return torch.FloatTensor(np.array(X))

def lstm_algorithm(dataframe_line:pd.DataFrame, pickle_rfc_predict_max_dir:str, pickle_rfc_predict_min_dir:str):

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
    
    dataframe_line = dataframe_line.reindex(columns=columns_order)

    with open('machine_learning_models/lstm/eurusd/pickle_files/lstm_regressor_scaler_x_max.pkl', 'rb') as file:
        scaler_x_max = pickle.load(file)
    
    with open('machine_learning_models/lstm/eurusd/pickle_files/lstm_regressor_scaler_y_max.pkl', 'rb') as file:
        scaler_y_max = pickle.load(file)

    with open('machine_learning_models/lstm/eurusd/pickle_files/lstm_regressor_scaler_x_min.pkl', 'rb') as file:
        scaler_x_min = pickle.load(file)
    
    with open('machine_learning_models/lstm/eurusd/pickle_files/lstm_regressor_scaler_y_min.pkl', 'rb') as file:
        scaler_y_min = pickle.load(file)

    scaled_data_x = scaler_x_max.fit_transform(dataframe_line)

    SEQ_LENGTH = 1

    X_data = create_sequences(scaled_data_x, SEQ_LENGTH)
    

    with open(pickle_rfc_predict_max_dir, 'rb') as file:
        model_lstm_predict_high = pickle.load(file)

    model_lstm_predict_high.eval()

    with torch.no_grad():
        y_pred = model_lstm_predict_high(X_data)

    high_value = scaler_y_max.inverse_transform(y_pred.numpy())

    with open(pickle_rfc_predict_min_dir, 'rb') as file1:
        model_lstm_predict_low = pickle.load(file1)

    model_lstm_predict_low.eval()
    with torch.no_grad():
        y_pred_low = model_lstm_predict_low(X_data)

    low_value = scaler_y_min.inverse_transform(y_pred_low.numpy())

    return high_value[-1][0], low_value[-1][0]
    
    
    