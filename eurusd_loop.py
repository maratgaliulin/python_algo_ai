from credentials import *
import time
import pandas as pd
from methods.make_dataframe_line import make_dataframe_line
from methods.buy_or_sell import buy_or_sell
from methods.save_to_csv import save_to_csv
# from machine_learning_models.random_forest.eurusd.random_forest_algorithm import random_forest_algorithm
# from machine_learning_models.xgboost.eurusd.xgboost_algorithm import xgboost_algorithm
# from machine_learning_models.lstm.eurusd.lstm_algorithm import lstm_algorithm
from machine_learning_models.lstm.eurusd.lstm_predict_candle_from_single_dataframe import predict_candle
from machine_learning_models.lstm.lstm_collect_predictions_into_dataframe import collect_predictions_into_dataframe
from methods.make_dataframe_from_server_for_training import make_dataframe_from_server_for_training
from methods.return_entry_point_and_takeprofit import return_entry_point_and_takeprofit

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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

ohlc_columns = ['open', 'high', 'low', 'close']

amount_of_30_second_intervals_in_a_day = 2880

interval_ordinal_number = 341

present_price_bid = mt.symbol_info_tick("EURUSD").bid

price_correction = present_price_bid * 0.9

while True:
    try:
        print('The number of 30-second interval is:', interval_ordinal_number)
        
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
                'BASE_DIR_LSTM': BASE_DIR_LSTM,
                'SAVED_DATAFRAME': SAVED_DATAFRAME,
                'SAVED_PREDICTIONS': SAVED_PREDICTIONS,
                'EURUSD_LSTM_MAX_VAL': EURUSD_LSTM_MAX_VAL,
                'EURUSD_LSTM_MIN_VAL': EURUSD_LSTM_MIN_VAL,
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
                'END_POSITION':92,
                'CORRECTION_INDEX':0.0,
                'COLUMNS_ORDER': COLUMNS_ORDER
            }
        
        
        
        if(interval_ordinal_number % amount_of_30_second_intervals_in_a_day == 0):

            price_correction = eurusd_dict['PRESENT_PRICE_BID'] * 0.9

            dataframe_for_training = make_dataframe_from_server_for_training(timeframe=eurusd_dict['TIMEFRAME_SMALL_MT'],
                                            start_pos=eurusd_dict['START_POSITION'],
                                            end_pos=300, columns_order=eurusd_dict['COLUMNS_ORDER'])
            
            for col in ohlc_columns:
                dataframe_for_training[col] = dataframe_for_training[col] - price_correction

            for col in columns_for_y:
                dataframe_for_training[col] = dataframe_for_training[col] - price_correction
            
            print(f'Length of the resulting training dataframe: {len(dataframe_for_training)}')

            
            for col in columns_for_y:
                print(f'Curreng y-column under training: {col}')
                predict_candle(df=dataframe_for_training, base_dir=BASE_DIR_LSTM, column_for_y=col, columns_order=eurusd_dict['COLUMNS_ORDER'])
        
        
        
        dataframe_line = make_dataframe_line(timeframe=eurusd_dict['TIMEFRAME_SMALL_MT'],
                                            start_pos=eurusd_dict['START_POSITION'],
                                            end_pos=eurusd_dict['END_POSITION'],
                                            columns_order=eurusd_dict['COLUMNS_ORDER']
                                            )
        
        # print('***************************')
        # print(dataframe_line)
        # print('***************************')
        
        for col in ohlc_columns:
            dataframe_line[col] = dataframe_line[col] - price_correction
                
        predicted_dataframe, _, high_value, low_value = collect_predictions_into_dataframe(dataframe_line=dataframe_line, base_dir_lstm=eurusd_dict['BASE_DIR_LSTM'], correction_index=eurusd_dict['CORRECTION_INDEX'], columns_order=eurusd_dict['COLUMNS_ORDER'])
        
        predicted_dataframe = predicted_dataframe + price_correction

        high_value += price_correction
        low_value += price_correction

        print(f"high and low values: {high_value}, {low_value}")

        save_to_csv(df_to_csv=dataframe_line[['open', 'high', 'low', 'close']], csv_address=eurusd_dict['SAVED_DATAFRAME'])
        save_to_csv(df_to_csv=predicted_dataframe, csv_address=eurusd_dict['SAVED_PREDICTIONS'])

        entry_point, take_profit, trend_direction = return_entry_point_and_takeprofit(
            predicted_high_price=high_value,
            predicted_low_price=low_value,
            present_price_bid=eurusd_dict['PRESENT_PRICE_BID'],
            present_price_ask=eurusd_dict['PRESENT_PRICE_ASK'],
            start_pos=eurusd_dict['START_POSITION'],
            end_pos=300
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
            entry_point=entry_point,
            take_profit=take_profit,
            trend_direction=trend_direction,
            magic=eurusd_dict['MAGIC']
        )

        time_sleep_total = eurusd_dict['SLEEP_TIME']

        print(f'Total sleep time: {time_sleep_total}')
        
        interval_ordinal_number += 1
        
        if(interval_ordinal_number > 2799):
            interval_ordinal_number = 0
            
        time.sleep(time_sleep_total)
    except Exception as e:
        print(e)
        
        interval_ordinal_number += 1
        
        time_sleep_total = eurusd_dict['SLEEP_TIME']

        print(f'Total sleep time: {time_sleep_total}')
        
        if(interval_ordinal_number > 2799):
            interval_ordinal_number = 0
            
        time.sleep(time_sleep_total)
        