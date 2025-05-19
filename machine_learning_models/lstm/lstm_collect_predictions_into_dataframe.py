import os 
import sys 

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(project_directory)

import pandas as pd

from .eurusd.lstm_use_prediction import use_prediction

def collect_predictions_into_dataframe(dataframe_line:pd.DataFrame, base_dir_lstm:str, correction_index:float) -> pd.DataFrame:

    columns_for_y = [
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
    open_values = pd.Series()
    high_values = pd.Series()
    low_values = pd.Series()
    close_values = pd.Series()

    last_candle_time = dataframe_line.iloc[-1].name

    num_future_candles = 8
    future_times = pd.date_range(
        start=last_candle_time + pd.Timedelta(minutes=5),
        periods=num_future_candles,
        freq='5min'
    ).tolist()
    
    for idx in range(len(columns_for_y)):
        complete_df_dir = base_dir_lstm + f'/lstm_regressor_predict_candle_{columns_for_y[idx]}.pkl'
        # print(idx % 4)
        
        predicted_value = use_prediction(dataframe_line=dataframe_line, 
                    predict_scaler_x=base_dir_lstm + '/lstm_regressor_scaler_x.pkl', 
                    predict_scaler_y=base_dir_lstm + f'/lstm_regressor_scaler_y_{columns_for_y[idx]}.pkl', 
                    y_predictor=base_dir_lstm + f'/lstm_regressor_predict_candle_{columns_for_y[idx]}.pkl')
        # print(predicted_value)
        if(idx % 4 == 0):
            open_values = pd.concat([open_values, pd.Series([predicted_value])], ignore_index=True)
        elif(idx % 4 == 1):
            high_values = pd.concat([high_values, pd.Series([predicted_value])], ignore_index=True)
        elif(idx % 4 == 2):
            low_values = pd.concat([low_values, pd.Series([predicted_value])], ignore_index=True)
        elif(idx % 4 == 3):
            close_values = pd.concat([close_values, pd.Series([predicted_value])], ignore_index=True)

    vals_dict = {
        'open': open_values,
        'high': high_values,
        'low': low_values,
        'close': close_values
    }

    predicted_values_dataframe = pd.DataFrame(vals_dict)

    predicted_values_dataframe.index = future_times
    
    high_value = predicted_values_dataframe['high'].max()
    low_value = predicted_values_dataframe['low'].min()

    return predicted_values_dataframe, high_value, low_value