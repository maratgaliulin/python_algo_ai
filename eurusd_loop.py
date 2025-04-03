from credentials import *
import time
import pandas as pd
from methods.make_dataframe_line import make_dataframe_line
from methods.buy_or_sell import buy_or_sell
# from machine_learning_models.random_forest.eurusd.random_forest_algorithm import random_forest_algorithm
from machine_learning_models.xgboost.eurusd.xgboost_algorithm import xgboost_algorithm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


while True:
     
    eurusd_dict = {
            'SYMBOL': "EURUSD",  # тикер валютной пары
            'BASE_DIR': BASE_DIR, # глубина зигзага
            'EURUSD_RFR_MAX_VAL_AFTER_TEST_TR': EURUSD_RFR_MAX_VAL_AFTER_TEST_TR,
            'EURUSD_RFR_MAX_VAL_BEFORE_TEST_TR': EURUSD_RFR_MAX_VAL_BEFORE_TEST_TR,
            'EURUSD_RFR_MIN_VAL_AFTER_TEST_TR': EURUSD_RFR_MIN_VAL_AFTER_TEST_TR,
            'EURUSD_RFR_MIN_VAL_BEFORE_TEST_TR': EURUSD_RFR_MIN_VAL_BEFORE_TEST_TR,
            'EURUSD_RFR_TREND_DIR_AFTER_TEST_TR': EURUSD_RFR_TREND_DIR_AFTER_TEST_TR,
            'EURUSD_RFR_TREND_DIR_BEFORE_TEST_TR': EURUSD_RFR_TREND_DIR_BEFORE_TEST_TR,
            'BASE_DIR_GRAD_BOOST': BASE_DIR_GRAD_BOOST, # глубина зигзага
            'EURUSD_GRAD_BOOST_MAX_VAL_AFTER_TEST_TR': EURUSD_GRAD_BOOST_MAX_VAL_AFTER_TEST_TR,
            'EURUSD_GRAD_BOOST_MAX_VAL_BEFORE_TEST_TR': EURUSD_GRAD_BOOST_MAX_VAL_BEFORE_TEST_TR,
            'EURUSD_GRAD_BOOST_MIN_VAL_AFTER_TEST_TR': EURUSD_GRAD_BOOST_MIN_VAL_AFTER_TEST_TR,
            'EURUSD_GRAD_BOOST_MIN_VAL_BEFORE_TEST_TR': EURUSD_GRAD_BOOST_MIN_VAL_BEFORE_TEST_TR,
            'EURUSD_GRAD_BOOST_TREND_DIR_AFTER_TEST_TR': EURUSD_GRAD_BOOST_TREND_DIR_AFTER_TEST_TR,
            'EURUSD_GRAD_BOOST_TREND_DIR_BEFORE_TEST_TR': EURUSD_GRAD_BOOST_TREND_DIR_BEFORE_TEST_TR,
            'SLEEP_TIME': sleep_time, #глубина поиска ордерблока
            'POINT': mt.symbol_info("EURUSD").point, # 1 пункт валютной пары
            'CURRENT_TIME': mt.symbol_info("EURUSD").time, # текущее время
            'PRESENT_PRICE_ASK': mt.symbol_info_tick("EURUSD").ask, # цена продажи на текущее время
            'PRESENT_PRICE_BID': mt.symbol_info_tick("EURUSD").bid, # цена покупки на текущее время
            'ORDERS_OF_THE_SYMBOL': mt.orders_get(symbol="EURUSD"), # наличие ордера данной валютной пары
            'POSITIONS_OF_THE_SYMBOL': mt.positions_get(symbol="EURUSD"), # наличие открытой позиции данной ВП
            'TIMEFRAME_SMALL_MT': TIMEFRAME_SMALL_MT, # короткий таймфрейм в формате MetaTrader5
            'LOT':0.1, # торговый объем
            'MAGIC': 52234337, # случайный номер, один из параметров для постановки лимитированного ордера. будет разным для различных валютных пар
            'ORDER_TYPE_BUY': ORDER_TYPE_BUY,
            'ORDER_TYPE_SELL':ORDER_TYPE_SELL,
            'ORDER_ACTION':ORDER_ACTION,
            'DETERMINE_THE_DEAL_ACTION':DETERMINE_THE_DEAL_ACTION,
            'CANCEL_THE_ORDER_ACTION': CANCEL_THE_ORDER_ACTION,
            'START_POSITION':0,
            'END_POSITION':13
            # 'CSV_ADDRESS': os.path.abspath(CSV_ADDRESS + "eurusd_order_blocks.csv"), # путь к папке с файлом, в котором записываются ордерблоки и их характеристики
            # 'FULL_CSV_PATH': os.path.abspath(PATH_TO_VARIABLES + "variables_eurusd.csv"), # путь к папке с файлом, в котором записываются переменные (главным образом для трейлинг СЛ)
            # 'ANALYSIS_LARGE_DATAFRAME': os.path.abspath(PATH_TO_ANALYSIS_DATAFRAMES + "/eurusd/ob_30_min_raw.csv"), # 
            # 'ANALYSIS_SMALL_DATAFRAME': os.path.abspath(PATH_TO_ANALYSIS_DATAFRAMES + "/eurusd/ob_1_min_raw.csv"), # 
            # 'ORDERS': os.path.abspath(PATH_TO_ANALYSIS_DATAFRAMES + "/eurusd/orders.csv"),
            # 'POSITIONS': os.path.abspath(PATH_TO_ANALYSIS_DATAFRAMES + "/eurusd/positions.csv"),
            # 'DEALS':os.path.abspath(PATH_TO_ANALYSIS_DATAFRAMES + "/eurusd/deals.csv"),
        }
    
    
    dataframe_line = make_dataframe_line(symbol=eurusd_dict['SYMBOL'],
                                         timeframe=eurusd_dict['TIMEFRAME_SMALL_MT'],
                                         start_pos=eurusd_dict['START_POSITION'],
                                         end_pos=eurusd_dict['END_POSITION']
                                         )
    
    # print(dataframe_line)
    
    # high_value, low_value, trend_direction = random_forest_algorithm(dataframe_line=dataframe_line,
    #                                                                  pickle_rfc_predict_max_dir=eurusd_dict['BASE_DIR'] + eurusd_dict['EURUSD_RFR_MAX_VAL_BEFORE_TEST_TR'],
    #                                                                  pickle_rfc_predict_min_dir=eurusd_dict['BASE_DIR'] + eurusd_dict['EURUSD_RFR_MIN_VAL_BEFORE_TEST_TR'],
    #                                                                  pickle_rfc_predict_trend_dir=eurusd_dict['BASE_DIR'] + eurusd_dict['EURUSD_RFR_TREND_DIR_BEFORE_TEST_TR']
    #                                                                  )
    
    high_value, low_value, trend_direction = xgboost_algorithm(dataframe_line=dataframe_line,
                                                                     pickle_rfc_predict_max_dir=eurusd_dict['BASE_DIR_GRAD_BOOST'] + eurusd_dict['EURUSD_GRAD_BOOST_MAX_VAL_AFTER_TEST_TR'],
                                                                     pickle_rfc_predict_min_dir=eurusd_dict['BASE_DIR_GRAD_BOOST'] + eurusd_dict['EURUSD_GRAD_BOOST_MIN_VAL_AFTER_TEST_TR'],
                                                                     pickle_rfc_predict_trend_dir=eurusd_dict['BASE_DIR_GRAD_BOOST'] + eurusd_dict['EURUSD_GRAD_BOOST_TREND_DIR_AFTER_TEST_TR']
                                                                     )
    
    time_sleep_modifier = buy_or_sell(
        min_impulse_size=0.0025,
        orders_of_the_symbol=eurusd_dict['ORDERS_OF_THE_SYMBOL'],
        positions_of_the_symbol=eurusd_dict['POSITIONS_OF_THE_SYMBOL'],
        present_price_bid=eurusd_dict['PRESENT_PRICE_BID'],
        present_price_ask=eurusd_dict['PRESENT_PRICE_ASK'],
        point=eurusd_dict['POINT'],
        lot=eurusd_dict['LOT'],
        order_type_buy=eurusd_dict['ORDER_TYPE_BUY'],
        order_type_sell=eurusd_dict['ORDER_TYPE_SELL'],
        order_action=eurusd_dict['ORDER_ACTION'],
        high_value=high_value[0],
        low_value=low_value[0],
        trend_direction=trend_direction,
        magic=eurusd_dict['MAGIC']
    )

    time_sleep_total = eurusd_dict['SLEEP_TIME'] + time_sleep_modifier

    print(f'Total sleep time: {time_sleep_total}')
    
    time.sleep(time_sleep_total)