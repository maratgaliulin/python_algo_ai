import pickle
import joblib
import dill
import pandas as pd
import numpy as np
import torch
# from sklearn.preprocessing import MinMaxScaler
from methods.generate_automatic_features_for_model import generate_automatic_features_for_model_test


# Примечания:

# 1. Как действовать в переходной зоной (переход от тренда к флэту) в скользящем окне

def create_sequences(data_x, seq_length):
    X = []
    for i in range(len(data_x) - seq_length):
        X.append(data_x[i:i+seq_length])  
    return torch.FloatTensor(np.array(X))

def use_prediction(dataframe_line:pd.DataFrame, predict_scaler_x:str, predict_scaler_y:str, y_predictor:str, columns_order:list, column_for_y:str, base_dir:str):
    
    production_ready_features = generate_automatic_features_for_model_test(df_raw=dataframe_line, cols_order=columns_order, base_dir=base_dir, column_for_y=column_for_y)

    columns_order_copy = columns_order.copy()

    with open(f'{base_dir}/feature_columns/feature_columns_{column_for_y}.pkl', 'rb') as file:
        feature_columns = dill.load(file)

    for col in feature_columns:
         columns_order_copy.append(col)
    
    dataframe_line = pd.merge(dataframe_line, production_ready_features, how="inner", left_index=True, right_index=True)

    dataframe_line = dataframe_line.reindex(columns=columns_order_copy)

    with open(predict_scaler_x, 'rb') as file:
        scaler_x = pickle.load(file)
    
    with open(predict_scaler_y, 'rb') as file:
        scaler_y = pickle.load(file)

    with open(y_predictor, 'rb') as file:
        model = pickle.load(file)

    scaled_data_x = scaler_x.fit_transform(dataframe_line)

    SEQ_LENGTH = 30

    X_data = create_sequences(scaled_data_x, SEQ_LENGTH)
    
    model.eval()

    # print('X_data:')
    # print(X_data)

    with torch.no_grad():
        y_pred = model(X_data)

    y_value = scaler_y.inverse_transform(y_pred.numpy())

    return y_value[-1][0]