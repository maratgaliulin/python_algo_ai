import pandas as pd
from .return_single_large_dataframe import return_single_large_dataframe
from .open_close_high_low_prices import return_open_price, return_close_price, return_high_price, return_low_price
from .adx_calculation import calculate_adx
from .define_the_trend import define_the_trend
import numpy as np
import os
import ta

def return_clean_dataframe(base_dir:str, time_series_folder:str, bid_or_ask_folder_bid:str, bid_or_ask_folder_ask:str):

    values = {
        'open_bid': 99999, 
        'high_bid': 99999, 
        'low_bid': 99999, 
        'close_bid': 99999, 
        'open_ask': 99999, 
        'high_ask': 99999, 
        'low_ask': 99999, 
        'close_ask': 99999,
        'volume_bid':0,
        'volume_ask':0
    }

    df_bid = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_bid)
    df_ask = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_ask)

    # print(f'{time_series_folder}:')
    # print(df_bid.head(10))
    # print(df_ask.head(10))

    df_joined = pd.merge(left=df_bid, right=df_ask, how='outer', sort=True, left_index=True, right_index=True, suffixes=['_bid', '_ask'])

    df_joined.fillna(inplace=True, value=values)

    # print(df_joined.loc[df_joined['open_bid'] == 999].head())

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

    df_joined.drop(['open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_bid', 'volume_ask'], inplace=True, axis=1)

    df_joined.dropna(inplace=True, axis=0)
    
    df_joined.sort_index(axis=1, ascending=True, inplace=True)

    print(df_joined.head(20))
    print(df_joined.tail(20))

    return df_joined

def make_single_df_from_bid_ask(base_dir:str, time_series_folder:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    columns_for_reindex = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "+DI",
        "-DI",
        "ADX",
        "ADL",
        "ATR_14",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_hist",
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
        "open_xauusd",
        "high_xauusd",
        "low_xauusd",
        "close_xauusd",
        "volume_xauusd",
        "open_jpyusd",
        "high_jpyusd",
        "low_jpyusd",
        "close_jpyusd",
        "volume_jpyusd",
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

        df_test_index = df_joined_eurusd.columns.to_list()
        print('list of indices:')
        print(df_test_index)
        print('\n\n\n\n')
        print('*****************************************')

        print('length of eurusd:', len(df_joined_eurusd))
        print('length of audusd:', len(df_joined_audusd))
        print('length of brentusd:', len(df_joined_brentusd))
        print('length of cadusd:', len(df_joined_cadusd))
        print('length of jpyusd:', len(df_joined_jpyusd))
        print('length of xauusd:', len(df_joined_xauusd))
        
        

        df_joined = pd.merge(df_joined_eurusd, df_joined_audusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_audusd'))
        df_joined = pd.merge(df_joined, df_joined_brentusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_brentusd'))
        df_joined = pd.merge(df_joined, df_joined_cadusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_cadusd'))
        df_joined = pd.merge(df_joined, df_joined_jpyusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_jpyusd'))
        df_joined = pd.merge(df_joined, df_joined_xauusd, how='outer', left_index=True, right_index=True, suffixes=(None, '_xauusd'))
        
        
        df_joined.sort_index(axis=1, ascending=True, inplace=True)

        
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

        
        
        


        df_joined = calculate_adx(df_joined, period=12)

        df_joined['RSI'] = ta.momentum.RSIIndicator(close=df_joined['close'], window=12).rsi()

        macd = ta.trend.MACD(close=df_joined['close'], window_slow=12, window_fast=4, window_sign=9)
        df_joined['MACD'] = macd.macd()  # MACD линия
        df_joined['MACD_signal'] = macd.macd_signal()  # Сигнальная линия
        df_joined['MACD_hist'] = macd.macd_diff()  # Гистограмма MACD

        # ADL (Accumulation/Distribution Line)
        df_joined['ADL'] = ta.volume.AccDistIndexIndicator(
            high=df_joined['high'],
            low=df_joined['low'],
            close=df_joined['close'],
            volume=df_joined['volume']
        ).acc_dist_index()

        df_joined['ATR_14'] = ta.volatility.AverageTrueRange(
            high=df_joined['high'],
            low=df_joined['low'],
            close=df_joined['close'],
            window=12  # Стандартный период для ATR
        ).average_true_range()

        sum_of_nans = df_joined.isna().sum()

        print('sum of nans: \n', sum_of_nans)

        
        # print(df_joined.loc[df_joined.isna().any(axis=1)].head(50))



        df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]

        

        df_len_60_percent = int(len(df_joined) * 0.6)
        df_len_80_percent = int(len(df_joined) * 0.8)
        
        df_joined_train = df_joined.iloc[0:df_len_60_percent]
        df_joined_test = df_joined.iloc[df_len_60_percent:df_len_80_percent]
        df_joined_val = df_joined.iloc[df_len_80_percent:]

        # print(df_joined_train.head())
        
        # print(len(df_joined))
        
        df_joined_train.sort_index(axis=1, ascending=True, inplace=True)
        df_joined_test.sort_index(axis=1, ascending=True, inplace=True)
        df_joined_val.sort_index(axis=1, ascending=True, inplace=True)

        df_joined_train = df_joined_train.loc[:, columns_for_reindex]
        df_joined_test = df_joined_test.loc[:, columns_for_reindex]
        df_joined_val = df_joined_val.loc[:, columns_for_reindex]

        # df_joined_train.to_csv(df_csv_file_train, index_label='time')
        # df_joined_test.to_csv(df_csv_file_test, index_label='time')
        # df_joined_val.to_csv(df_csv_file_validation, index_label='time')

        return df_joined_train, df_joined_test, df_joined_val
        # return df_join