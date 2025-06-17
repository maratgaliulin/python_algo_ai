import MetaTrader5 as mt
from account_info import *


if not mt.initialize():
    print("initialize() failed, error code =", mt.last_error())
    quit()


authorized = mt.login(account, pw)


ohlc_columns = ['open', 'high', 'low', 'close']

amount_of_30_second_intervals_in_a_day = 2880

# COMMON VARIABLES 

sleep_time = 1

BASE_DIR = 'machine_learning_models/random_forest/eurusd/pickle_files'

# EURUSD PICKLE FILES DIRECTORIES

EURUSD_RFR_MAX_VAL_AFTER_TEST_TR = '/random_forest/random_forest_regressor_predict_max_values_after_test_training.pkl'
EURUSD_RFR_MAX_VAL_BEFORE_TEST_TR = '/random_forest/random_forest_regressor_predict_max_values-2025-03-25.pkl'

EURUSD_RFR_MIN_VAL_AFTER_TEST_TR = '/random_forest/random_forest_regressor_predict_min_values_after_test_training.pkl'
EURUSD_RFR_MIN_VAL_BEFORE_TEST_TR = '/random_forest/random_forest_regressor_predict_min_values-2025-03-25.pkl'

EURUSD_RFR_TREND_DIR_AFTER_TEST_TR = '/random_forest/random_forest_regressor_predict_trend_direction_after_test_training.pkl'
EURUSD_RFR_TREND_DIR_BEFORE_TEST_TR = '/random_forest/random_forest_regressor_predict_trend_direction-2025-03-25.pkl'


BASE_DIR_GRAD_BOOST = 'machine_learning_models/xgboost/eurusd/pickle_files'

EURUSD_GRAD_BOOST_MAX_VAL_AFTER_TEST_TR = '/xgboost_regressor_predict_max_values_after_test_training-2025-03-27.pkl'
EURUSD_GRAD_BOOST_MAX_VAL_BEFORE_TEST_TR = '/xgboost_regressor_predict_max_values-2025-03-27.pkl'

EURUSD_GRAD_BOOST_MIN_VAL_AFTER_TEST_TR = '/xgboost_regressor_predict_min_values_after_test_training-2025-03-27.pkl'
EURUSD_GRAD_BOOST_MIN_VAL_BEFORE_TEST_TR = '/xgboost_regressor_predict_min_values-2025-03-27.pkl'

EURUSD_GRAD_BOOST_TREND_DIR_AFTER_TEST_TR = '/xgboost_regressor_predict_trend_direction_after_test_training-2025-03-27.pkl'
EURUSD_GRAD_BOOST_TREND_DIR_BEFORE_TEST_TR = '/xgboost_regressor_predict_trend_direction-2025-03-27.pkl'

BASE_DIR_LSTM = 'machine_learning_models/lstm/eurusd/pickle_files'

EURUSD_LSTM_MAX_VAL = '/lstm_regressor_predict_max_values-2025-04-07.pkl'

EURUSD_LSTM_MIN_VAL = '/lstm_regressor_predict_min_values-2025-04-07.pkl'

SAVED_DATAFRAME = 'machine_learning_models/lstm/eurusd/saved_dataframe.csv'
SAVED_PREDICTIONS = 'machine_learning_models/lstm/eurusd/saved_predictions.csv'


# BUY / SELL_ORDER_BLOCK

ORDER_TYPE_BUY = mt.ORDER_TYPE_BUY_LIMIT
ORDER_TYPE_SELL = mt.ORDER_TYPE_SELL_LIMIT
ORDER_ACTION = mt.TRADE_ACTION_PENDING

# DETERMINE_THE_DEAL

DETERMINE_THE_DEAL_ACTION = mt.TRADE_ACTION_PENDING

# CANCEL THE ORDER

CANCEL_THE_ORDER_ACTION = mt.TRADE_ACTION_REMOVE

# TIMEFRAMES

TIMEFRAME_SMALL = '5min'
TIMEFRAME_SMALL_MT = mt.TIMEFRAME_M5

COLUMNS_ORDER = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ADX",
        "ADL",
        "ATR_14",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        '40_min_vol',
        '20_min_ma',
        'day_of_week',
        'is_month_end',
        'volume_ma_20_min',
        'volume_ma_ratio',
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
