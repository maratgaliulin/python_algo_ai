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

def use_prediction(dataframe_line:pd.DataFrame, predict_scaler_x:str, predict_scaler_y:str, y_predictor:str):

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
    
    dataframe_line = dataframe_line.reindex(columns=columns_order)

    with open(predict_scaler_x, 'rb') as file:
        scaler_x = pickle.load(file)
    
    with open(predict_scaler_y, 'rb') as file:
        scaler_y = pickle.load(file)

    with open(y_predictor, 'rb') as file:
        model = pickle.load(file)

    scaled_data_x = scaler_x.fit_transform(dataframe_line)

    SEQ_LENGTH = 60

    X_data = create_sequences(scaled_data_x, SEQ_LENGTH)

    model.eval()

    with torch.no_grad():
        y_pred = model(X_data)

    y_value = scaler_y.inverse_transform(y_pred.numpy())

    return y_value[-1][0]