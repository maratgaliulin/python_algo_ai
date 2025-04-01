import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Must be before torch import!
import torch

import pandas as pd
# from methods.draw_graph import draw_static_graph
from methods.make_single_df_from_bid_ask import make_single_df_from_bid_ask

# from machine_learning_models.random_forest.eurusd.random_forest_regression_predict_max_value import search_optimal_parameters_for_random_forest_max_value_prediction, predict_max_value_with_random_forest_regressor
# from machine_learning_models.random_forest.eurusd.random_forest_regression_predict_min_values import search_optimal_parameters_for_random_forest_min_value_prediction, predict_min_value_with_random_forest_regressor
# from machine_learning_models.random_forest.eurusd.random_forest_classifier_predict_trend_direction import predict_trend_direction_with_random_forest_classifier

# from machine_learning_models.xgboost.eurusd.xgboost_classifier_predict_trend_direction import predict_trend_direction_with_gradient_boost_classifier
# from machine_learning_models.xgboost.eurusd.xgboost_regression_predict_max_value import predict_max_value_with_gradient_boost_regressor
# from machine_learning_models.xgboost.eurusd.xgboost_regression_predict_min_values import predict_min_value_with_gradient_boost_regressor

from machine_learning_models.lstm.eurusd.lstm_predict_max_value import predict_max_value_with_lstm_model

# from methods.make_single_df_test import make_single_df_from_bid_test


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# base_dir = "hist_data/EURUSD/1_min/"

# dir_bid = base_dir + "Bid/EURUSD_Candlestick_1_m_BID_01.01.2004-01.01.2007.csv"
# dir_ask = base_dir + "Ask/EURUSD_Candlestick_1_M_ASK_01.01.2004-01.01.2007.csv"


# df_5min_bid = pd.read_csv(dir_bid, index_col="Gmt time").sort_index(ascending=True)
# df_5min_ask = pd.read_csv(dir_ask, index_col="Gmt time").sort_index(ascending=True)

# df_joined = make_single_df_from_bid_test(df_bid=df_5min_bid, df_ask=df_5min_ask)

# print(df_joined.head())



# print(df_5min_bid.tail(50))
# print('********************')
# print(df_5min_ask.tail(50))
# print('********************')
# for i in range(0,90):
#     print(df_5min_joined.iloc[i].T)

# draw_static_graph(df_5min_bid.tail(70), df_5min_ask.tail(70), df_5min_joined.tail(70))



bdir = "hist_data/"
time_series_folder = "EURUSD/"
bid_or_ask_folder_bid = "5_min/Bid/"
bid_or_ask_folder_ask = "5_min/Ask/"

df_5min_joined_train, df_5min_joined_test, df_5min_joined_val = make_single_df_from_bid_ask(
    base_dir=bdir, 
    time_series_folder=time_series_folder, 
    bid_or_ask_folder_bid=bid_or_ask_folder_bid, 
    bid_or_ask_folder_ask=bid_or_ask_folder_ask
    )

# print(df_5min_joined_train.tail(1))
# print('*************************')
# print(df_5min_joined_test.head(1))
# print('*************************')
# print(df_5min_joined_test.tail(1))
# print('*************************')
# print(df_5min_joined_val.head(1))


# print(len(df_5min_joined_train))
# print(len(df_5min_joined_test))
# print(len(df_5min_joined_val))
# print()

random_forest_base_dir_algo = 'machine_learning_models/random_forest/eurusd/pickle_files'

gradient_boost_base_dir_algo = 'machine_learning_models/xgboost/eurusd/pickle_files'

lstm_base_dir_algo = 'machine_learning_models/lstm/eurusd/pickle_files'

# search_optimal_parameters_for_random_forest_max_value_prediction(df_5min_joined_train)

# search_optimal_parameters_for_random_forest_min_value_prediction(df_5min_joined_train)

# predict_trend_direction_with_random_forest_classifier(df_5min_joined, base_dir_algo)

# predict_max_value_with_random_forest_regressor(df_5min_joined, base_dir_algo)

# predict_min_value_with_random_forest_regressor(df_5min_joined, base_dir_algo)

# print(df_5min_joined_train.head(10))


# predict_trend_direction_with_gradient_boost_classifier(df_5min_joined_train, df_5min_joined_test, df_5min_joined_val, gradient_boost_base_dir_algo)

# predict_min_value_with_gradient_boost_regressor(df_5min_joined_train, df_5min_joined_test, df_5min_joined_val, gradient_boost_base_dir_algo)

# predict_max_value_with_gradient_boost_regressor(df_5min_joined_train, df_5min_joined_test, df_5min_joined_val, gradient_boost_base_dir_algo)


# predict_max_value_with_lstm_model(df_5min_joined_train, df_5min_joined_test, df_5min_joined_val, lstm_base_dir_algo)


# print('uptrend', 'downtrend', 'undefined')

# print(df_5min_joined['y_trend_uptrend'].sum(), df_5min_joined['y_trend_downtrend'].sum(), df_5min_joined['y_trend_trend undefined'].sum())

# print(df_5min_joined_train.columns)

# with open(base_dir_algo, 'rb') as file:
#     model_random_forest_predict_trend = pickle.load(file)
    
# df_5min_joined_val = df_5min_joined_val.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    
# y_predict = model_random_forest_predict_trend.predict(df_5min_joined_val)

# print(y_predict[0])