import pandas as pd
from .return_single_large_dataframe import return_single_large_dataframe
from .open_close_high_low_prices import return_open_price, return_close_price, return_high_price, return_low_price
from .adx_calculation import calculate_adx
from .define_the_trend import define_the_trend
import numpy as np
import os

def return_clean_dataframe(base_dir:str, time_series_folder:str, bid_or_ask_folder_bid:str, bid_or_ask_folder_ask:str):

    values = {
        'open_bid': 999, 
        'high_bid': 999, 
        'low_bid': 999, 
        'close_bid': 999, 
        'open_ask': 999, 
        'high_ask': 999, 
        'low_ask': 999, 
        'close_ask': 999,
        'volume_bid':0,
        'volume_ask':0
    }

    rename_columns_dict = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
    }

    df_bid = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_bid)
    df_ask = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_ask)

    df_bid.rename(columns=rename_columns_dict, inplace=True)
    df_ask.rename(columns=rename_columns_dict, inplace=True)

    df_joined = pd.merge(df_bid, df_ask, how='outer', left_index=True, right_index=True, suffixes=['_bid', '_ask'])

    df_joined.fillna(inplace=True, value=values)

    df_joined['open'] = df_joined.apply(
        lambda row: return_open_price(row, open_bid='open_bid', open_ask='open_ask', close_bid='close_bid'), 
        axis=1
        )
    df_joined['close'] = df_joined.apply(
        lambda row: return_close_price(row, close_bid='close_bid', close_ask='close_ask', open_bid='open_bid', open_ask='open_ask'), 
        axis=1
        )
    df_joined['high'] = df_joined.apply(
        lambda row: return_high_price(row, high_bid='high_bid', high_ask='high_ask'), 
        axis=1
        )
    df_joined['low'] = df_joined.apply(
        lambda row: return_low_price(row, low_bid='low_bid', low_ask='low_ask'), 
        axis=1
        )
    df_joined['volume'] = df_joined['volume_bid'] + df_joined['volume_ask']

    df_joined.drop(['open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_bid', 'volume_ask', 'Gmt time_bid', 'Gmt time_ask'], inplace=True, axis=1)

    df_joined = calculate_adx(df_joined, period=12)

    # df_joined.sort_index(inplace=True)

    return df_joined

def make_single_df_from_bid_ask(base_dir:str, time_series_folder:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = {
        'open_bid': 999, 
        'high_bid': 999, 
        'low_bid': 999, 
        'close_bid': 999, 
        'open_ask': 999, 
        'high_ask': 999, 
        'low_ask': 999, 
        'close_ask': 999,
        'volume_bid':0,
        'volume_ask':0
    }

    rename_columns_dict = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
    }
    
    columns_for_y = ['high', 
     'high_plus_5min', 
     'high_plus_10min', 
     'high_plus_15min',
     'high_plus_20min',
     'high_plus_25min',
     'high_plus_30min',
     'high_plus_35min',
     'high_plus_40min',
     'high_plus_45min',
     'high_plus_50min',
     'high_plus_55min',
     'high_plus_60min',
     ]
    
    complete_df_dir = base_dir + time_series_folder + 'complete/'
    
    df_csv_file_train = complete_df_dir + 'df_file_train.csv'
    df_csv_file_test = complete_df_dir + 'df_file_test.csv'
    df_csv_file_validation = complete_df_dir + 'df_file_validation.csv'
    
    complete_files_exist = os.path.isfile(df_csv_file_train) and os.path.isfile(df_csv_file_test) and os.path.isfile(df_csv_file_validation)
    
    if complete_files_exist:
        df_joined_train = pd.read_csv(df_csv_file_train, index_col=['time'])
        df_joined_test = pd.read_csv(df_csv_file_test, index_col=['time'])
        df_joined_val = pd.read_csv(df_csv_file_validation, index_col=['time'])
        
        df_joined_train.sort_index(axis=1, ascending=True, inplace=True)
        df_joined_test.sort_index(axis=1, ascending=True, inplace=True)
        df_joined_val.sort_index(axis=1, ascending=True, inplace=True)
        
        df_joined_train.index = pd.to_datetime(df_joined_train.index, format='mixed')
        df_joined_test.index = pd.to_datetime(df_joined_test.index, format='mixed')
        df_joined_val.index = pd.to_datetime(df_joined_val.index, format='mixed')
        
        return df_joined_train, df_joined_test, df_joined_val
    
    else:

        BASE_DIR = 'hist_data/'

        time_series_eurusd = 'EURUSD/'
        time_series_audusd = 'AUDUSD/'
        time_series_brentusd = 'BrentUSD/'
        time_series_cadusd = 'CADUSD/'
        time_series_jpyusd = 'JPYUSD/'
        time_series_xauusd = 'XAUUSD/'
        
        BID_FOLDER = '5_min/Bid/'
        ASK_FOLDER = '5_min/Ask/'

        df_joined_eurusd = return_clean_dataframe(base_dir=BASE_DIR, time_series_folder=time_series_eurusd, bid_or_ask_folder_bid=BID_FOLDER, bid_or_ask_folder_ask=ASK_FOLDER)
        df_joined_audusd = return_clean_dataframe(base_dir=BASE_DIR, time_series_folder=time_series_audusd, bid_or_ask_folder_bid=BID_FOLDER, bid_or_ask_folder_ask=ASK_FOLDER)
        df_joined_brentusd = return_clean_dataframe(base_dir=BASE_DIR, time_series_folder=time_series_brentusd, bid_or_ask_folder_bid=BID_FOLDER, bid_or_ask_folder_ask=ASK_FOLDER)
        df_joined_cadusd = return_clean_dataframe(base_dir=BASE_DIR, time_series_folder=time_series_cadusd, bid_or_ask_folder_bid=BID_FOLDER, bid_or_ask_folder_ask=ASK_FOLDER)
        df_joined_jpyusd = return_clean_dataframe(base_dir=BASE_DIR, time_series_folder=time_series_jpyusd, bid_or_ask_folder_bid=BID_FOLDER, bid_or_ask_folder_ask=ASK_FOLDER)
        df_joined_xauusd = return_clean_dataframe(base_dir=BASE_DIR, time_series_folder=time_series_xauusd, bid_or_ask_folder_bid=BID_FOLDER, bid_or_ask_folder_ask=ASK_FOLDER)
        
        

        df_joined = pd.merge(df_joined_eurusd, df_joined_audusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_audusd'))
        df_joined = pd.merge(df_joined, df_joined_brentusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_brentusd'))
        df_joined = pd.merge(df_joined, df_joined_cadusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_cadusd'))
        df_joined = pd.merge(df_joined, df_joined_jpyusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_jpyusd'))
        df_joined = pd.merge(df_joined, df_joined_xauusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_xauusd'))

        
        df_joined['open_plus_5min'] = df_joined['open'].shift(-1)
        df_joined['high_plus_5min'] = df_joined['high'].shift(-1)
        df_joined['low_plus_5min'] = df_joined['low'].shift(-1)
        df_joined['close_plus_5min'] = df_joined['close'].shift(-1)

        df_joined['open_plus_10min'] = df_joined['open'].shift(-2)
        df_joined['high_plus_10min'] = df_joined['high'].shift(-2)
        df_joined['low_plus_10min'] = df_joined['low'].shift(-2)
        df_joined['close_plus_10min'] = df_joined['close'].shift(-2)

        df_joined['open_plus_15min'] = df_joined['open'].shift(-3)
        df_joined['high_plus_15min'] = df_joined['high'].shift(-3)
        df_joined['low_plus_15min'] = df_joined['low'].shift(-3)
        df_joined['close_plus_15min'] = df_joined['close'].shift(-3)

        df_joined['open_plus_20min'] = df_joined['open'].shift(-4)
        df_joined['high_plus_20min'] = df_joined['high'].shift(-4)
        df_joined['low_plus_20min'] = df_joined['low'].shift(-4)
        df_joined['close_plus_20min'] = df_joined['close'].shift(-4)

        df_joined['open_plus_25min'] = df_joined['open'].shift(-5)
        df_joined['high_plus_25min'] = df_joined['high'].shift(-5)
        df_joined['low_plus_25min'] = df_joined['low'].shift(-5)
        df_joined['close_plus_25min'] = df_joined['close'].shift(-5)

        df_joined['open_plus_30min'] = df_joined['open'].shift(-6)
        df_joined['high_plus_30min'] = df_joined['high'].shift(-6)
        df_joined['low_plus_30min'] = df_joined['low'].shift(-6)
        df_joined['close_plus_30min'] = df_joined['close'].shift(-6)

        df_joined['open_plus_35min'] = df_joined['open'].shift(-7)
        df_joined['high_plus_35min'] = df_joined['high'].shift(-7)
        df_joined['low_plus_35min'] = df_joined['low'].shift(-7)
        df_joined['close_plus_35min'] = df_joined['close'].shift(-7)

        df_joined['open_plus_40min'] = df_joined['open'].shift(-8)
        df_joined['high_plus_40min'] = df_joined['high'].shift(-8)
        df_joined['low_plus_40min'] = df_joined['low'].shift(-8)
        df_joined['close_plus_40min'] = df_joined['close'].shift(-8)


        
        
        
         
        # df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]

        df_len_60_percent = int(len(df_joined) * 0.6)
        df_len_80_percent = int(len(df_joined) * 0.8)
        
        df_joined_train = df_joined.iloc[0:df_len_60_percent]
        df_joined_test = df_joined.iloc[df_len_60_percent:df_len_80_percent]
        df_joined_val = df_joined.iloc[df_len_80_percent:]

        # print(df_joined_train.head())
        
        # print(len(df_joined))
        
        # df_joined_train.sort_index(inplace=True)
        # df_joined_test.sort_index(inplace=True)
        # df_joined_val.sort_index(inplace=True)

        # df_joined_train.to_csv(df_csv_file_train, index_label='time')
        # df_joined_test.to_csv(df_csv_file_test, index_label='time')
        # df_joined_val.to_csv(df_csv_file_validation, index_label='time')

        return df_joined_train, df_joined_test, df_joined_val
        # return df_join