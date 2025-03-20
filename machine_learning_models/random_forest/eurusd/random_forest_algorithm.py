import pickle
import pandas as pd

def random_forest_algorithm(dataframe_line:pd.DataFrame, pickle_rfc_predict_max_dir:str, pickle_rfc_predict_min_dir:str, pickle_rfc_predict_trend_dir:str):
    with open(pickle_rfc_predict_max_dir, 'rb') as file:
        model_random_forest_predict_high = pickle.load(file)

    with open(pickle_rfc_predict_min_dir, 'rb') as file1:
        model_random_forest_predict_low = pickle.load(file1)

    with open(pickle_rfc_predict_trend_dir, 'rb') as file2:
        model_random_forest_predict_trend_direction = pickle.load(file2)


    high_value = model_random_forest_predict_high.predict(dataframe_line)
    low_value = model_random_forest_predict_low.predict(dataframe_line)
    trend_direction = model_random_forest_predict_trend_direction.predict(dataframe_line)
    # trend_direction = trend_direction.idxmax(axis=1)

    return high_value, low_value, trend_direction
    
    
    