import MetaTrader5 as mt

if not mt.initialize():
    print("initialize() failed, error code =", mt.last_error())
    quit()

account=52234337
pw = 'b&89zG&!PruWtP'
authorized = mt.login(account, pw)



# COMMON VARIABLES 

sleep_time = 5

CSV_ADDRESS = "order_block_csv_files/"

PATH_TO_VARIABLES = "order_block/variables/"

PATH_TO_ANALYSIS_DATAFRAMES = "dataframes_for_analysis/"

# BUY / SELL_ORDER_BLOCK

ORDER_TYPE_BUY = mt.ORDER_TYPE_BUY_LIMIT
ORDER_TYPE_SELL = mt.ORDER_TYPE_SELL_LIMIT
ORDER_ACTION = mt.TRADE_ACTION_PENDING

# DETERMINE_THE_DEAL

DETERMINE_THE_DEAL_ACTION = mt.TRADE_ACTION_PENDING

# CANCEL THE ORDER

CANCEL_THE_ORDER_ACTION = mt.TRADE_ACTION_REMOVE

# TIMEFRAMES

TIMEFRAME_SMALL = ['1min', '5min']
TIMEFRAME_SMALL_MT = [
    mt.TIMEFRAME_M1,
    mt.TIMEFRAME_M5
]
