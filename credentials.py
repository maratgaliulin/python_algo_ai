import MetaTrader5 as mt

if not mt.initialize():
    print("initialize() failed, error code =", mt.last_error())
    quit()

account=52234337
pw = 'b&89zG&!PruWtP'
authorized = mt.login(account, pw)



# COMMON VARIABLES 

sleep_time = 30

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
