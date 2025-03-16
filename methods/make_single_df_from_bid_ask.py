import pandas as pd
from .return_single_large_dataframe import return_single_large_dataframe
from .open_close_high_low_prices import return_open_price, return_close_price, return_high_price, return_low_price
from .adx_calculation import calculate_adx
import numpy as np


def define_the_trend(df:pd.DataFrame):
    highs_less_than_zero = df['high_0min_60min_difference'] < 0
    lows_less_than_zero = df['low_0min_60min_difference'] < 0
    
    highs_more_than_zero = df['high_0min_60min_difference'] > 0
    lows_more_than_zero = df['low_0min_60min_difference'] > 0
    
    if(highs_less_than_zero and lows_less_than_zero):
        return 'uptrend'
    elif(highs_more_than_zero and lows_more_than_zero):
        return 'downtrend'
    else:
        return 'trend undefined'


def make_single_df_from_bid_ask(base_dir:str, time_series_folder:str, bid_or_ask_folder_bid:str, bid_or_ask_folder_ask:str) -> pd.DataFrame:
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
    
    columns_for_y_60min_max = ['high', 
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
    
    columns_for_y_60min_min = ['low', 
     'low_plus_5min', 
     'low_plus_10min', 
     'low_plus_15min',
     'low_plus_20min',
     'low_plus_25min',
     'low_plus_30min',
     'low_plus_35min',
     'low_plus_40min',
     'low_plus_45min',
     'low_plus_50min',
     'low_plus_55min',
     'low_plus_60min',
     ]

    df_bid = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_bid)
    df_ask = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_ask)
    
    df_bid.rename(columns=rename_columns_dict, inplace=True)
    df_ask.rename(columns=rename_columns_dict, inplace=True)

    df_joined = pd.merge(df_bid, df_ask, how='outer', left_index=True, right_index=True, suffixes=['_bid', '_ask'])
    df_joined.fillna(inplace=True, value=values)
    df_joined['open'] = df_joined.apply(return_open_price, axis=1)
    df_joined['open_minus_5min'] = df_joined['open'].shift(1)
    df_joined['open_minus_10min'] = df_joined['open_minus_5min'].shift(1)
    df_joined['open_minus_15min'] = df_joined['open_minus_10min'].shift(1)
    df_joined['open_minus_20min'] = df_joined['open_minus_15min'].shift(1)
    df_joined['open_minus_25min'] = df_joined['open_minus_20min'].shift(1)
    df_joined['open_minus_30min'] = df_joined['open_minus_25min'].shift(1)
    df_joined['open_minus_35min'] = df_joined['open_minus_30min'].shift(1)
    df_joined['open_minus_40min'] = df_joined['open_minus_35min'].shift(1)
    df_joined['open_minus_45min'] = df_joined['open_minus_40min'].shift(1)
    df_joined['open_minus_50min'] = df_joined['open_minus_45min'].shift(1)
    df_joined['open_minus_55min'] = df_joined['open_minus_50min'].shift(1)
    df_joined['open_minus_60min'] = df_joined['open_minus_55min'].shift(1)
    
    df_joined['close'] = df_joined.apply(return_close_price, axis=1)
    df_joined['close_minus_5min'] = df_joined['close'].shift(1)
    df_joined['close_minus_10min'] = df_joined['close_minus_5min'].shift(1)
    df_joined['close_minus_15min'] = df_joined['close_minus_10min'].shift(1)
    df_joined['close_minus_20min'] = df_joined['close_minus_15min'].shift(1)
    df_joined['close_minus_25min'] = df_joined['close_minus_20min'].shift(1)
    df_joined['close_minus_30min'] = df_joined['close_minus_25min'].shift(1)
    df_joined['close_minus_35min'] = df_joined['close_minus_30min'].shift(1)
    df_joined['close_minus_40min'] = df_joined['close_minus_35min'].shift(1)
    df_joined['close_minus_45min'] = df_joined['close_minus_40min'].shift(1)
    df_joined['close_minus_50min'] = df_joined['close_minus_45min'].shift(1)
    df_joined['close_minus_55min'] = df_joined['close_minus_50min'].shift(1)
    df_joined['close_minus_60min'] = df_joined['close_minus_55min'].shift(1)
    
    df_joined['high'] = df_joined.apply(return_high_price, axis=1)
    df_joined['high_minus_5min'] = df_joined['high'].shift(1)
    df_joined['high_minus_10min'] = df_joined['high_minus_5min'].shift(1)
    df_joined['high_minus_15min'] = df_joined['high_minus_10min'].shift(1)
    df_joined['high_minus_20min'] = df_joined['high_minus_15min'].shift(1)
    df_joined['high_minus_25min'] = df_joined['high_minus_20min'].shift(1)
    df_joined['high_minus_30min'] = df_joined['high_minus_25min'].shift(1)
    df_joined['high_minus_35min'] = df_joined['high_minus_30min'].shift(1)
    df_joined['high_minus_40min'] = df_joined['high_minus_35min'].shift(1)
    df_joined['high_minus_45min'] = df_joined['high_minus_40min'].shift(1)
    df_joined['high_minus_50min'] = df_joined['high_minus_45min'].shift(1)
    df_joined['high_minus_55min'] = df_joined['high_minus_50min'].shift(1)
    df_joined['high_minus_60min'] = df_joined['high_minus_55min'].shift(1)
    
    df_joined['low'] = df_joined.apply(return_low_price, axis=1)
    df_joined['low_minus_5min'] = df_joined['low'].shift(1)
    df_joined['low_minus_10min'] = df_joined['low_minus_5min'].shift(1)
    df_joined['low_minus_15min'] = df_joined['low_minus_10min'].shift(1)
    df_joined['low_minus_20min'] = df_joined['low_minus_15min'].shift(1)
    df_joined['low_minus_25min'] = df_joined['low_minus_20min'].shift(1)
    df_joined['low_minus_30min'] = df_joined['low_minus_25min'].shift(1)
    df_joined['low_minus_35min'] = df_joined['low_minus_30min'].shift(1)
    df_joined['low_minus_40min'] = df_joined['low_minus_35min'].shift(1)
    df_joined['low_minus_45min'] = df_joined['low_minus_40min'].shift(1)
    df_joined['low_minus_50min'] = df_joined['low_minus_45min'].shift(1)
    df_joined['low_minus_55min'] = df_joined['low_minus_50min'].shift(1)
    df_joined['low_minus_60min'] = df_joined['low_minus_55min'].shift(1)
    
    df_joined['volume'] = df_joined['volume_bid'] + df_joined['volume_ask']
    df_joined['volume_minus_5min'] = df_joined['volume'].shift(1)
    df_joined['volume_minus_10min'] = df_joined['volume_minus_5min'].shift(1)
    df_joined['volume_minus_15min'] = df_joined['volume_minus_10min'].shift(1)
    df_joined['volume_minus_20min'] = df_joined['volume_minus_15min'].shift(1)
    df_joined['volume_minus_25min'] = df_joined['volume_minus_20min'].shift(1)
    df_joined['volume_minus_30min'] = df_joined['volume_minus_25min'].shift(1)
    df_joined['volume_minus_35min'] = df_joined['volume_minus_30min'].shift(1)
    df_joined['volume_minus_40min'] = df_joined['volume_minus_35min'].shift(1)
    df_joined['volume_minus_45min'] = df_joined['volume_minus_40min'].shift(1)
    df_joined['volume_minus_50min'] = df_joined['volume_minus_45min'].shift(1)
    df_joined['volume_minus_55min'] = df_joined['volume_minus_50min'].shift(1)
    df_joined['volume_minus_60min'] = df_joined['volume_minus_55min'].shift(1)
    
    df_joined_open_mean = df_joined['open'].mean()
    df_joined_close_mean = df_joined['close'].mean()
    df_joined_high_mean = df_joined['high'].mean()
    df_joined_low_mean = df_joined['low'].mean()
    df_joined_volume_mean = df_joined['volume'].mean()
    
    df_joined_open_std = df_joined['open'].std()
    df_joined_close_std = df_joined['close'].std()
    df_joined_high_std = df_joined['high'].std()
    df_joined_low_std = df_joined['low'].std()
    df_joined_volume_std = df_joined['volume'].std()
    
    df_joined['open_normalized'] = (df_joined['open'] - df_joined_open_mean)/df_joined_open_std
    df_joined['close_normalized'] = (df_joined['close'] - df_joined_close_mean)/df_joined_close_std
    df_joined['high_normalized'] = (df_joined['high'] - df_joined_high_mean)/df_joined_high_std
    df_joined['low_normalized'] = (df_joined['low'] - df_joined_low_mean)/df_joined_low_std
    df_joined['volume_normalized'] = (df_joined['volume'] - df_joined_volume_mean)/df_joined_volume_std
    
    df_joined['open_log'] = np.log(df_joined['open'])
    df_joined['close_log'] = np.log(df_joined['close'])
    df_joined['high_log'] = np.log(df_joined['high'])
    df_joined['low_log'] = np.log(df_joined['low'])
    df_joined['volume_log'] = np.log(df_joined['volume'])
    
    df_joined = calculate_adx(df_joined, period=12)
    
    df_joined['high_plus_5min'] = df_joined['high'].shift(-1)
    df_joined['high_plus_10min'] = df_joined['high_plus_5min'].shift(-1)
    df_joined['high_plus_15min'] = df_joined['high_plus_10min'].shift(-1)
    df_joined['high_plus_20min'] = df_joined['high_plus_15min'].shift(-1)
    df_joined['high_plus_25min'] = df_joined['high_plus_20min'].shift(-1)
    df_joined['high_plus_30min'] = df_joined['high_plus_25min'].shift(-1)
    df_joined['high_plus_35min'] = df_joined['high_plus_30min'].shift(-1)
    df_joined['high_plus_40min'] = df_joined['high_plus_35min'].shift(-1)
    df_joined['high_plus_45min'] = df_joined['high_plus_40min'].shift(-1)
    df_joined['high_plus_50min'] = df_joined['high_plus_45min'].shift(-1)
    df_joined['high_plus_55min'] = df_joined['high_plus_50min'].shift(-1)
    df_joined['high_plus_60min'] = df_joined['high_plus_55min'].shift(-1)
    
    df_joined['low_plus_5min'] = df_joined['low'].shift(-1)
    df_joined['low_plus_10min'] = df_joined['low_plus_5min'].shift(-1)
    df_joined['low_plus_15min'] = df_joined['low_plus_10min'].shift(-1)
    df_joined['low_plus_20min'] = df_joined['low_plus_15min'].shift(-1)
    df_joined['low_plus_25min'] = df_joined['low_plus_20min'].shift(-1)
    df_joined['low_plus_30min'] = df_joined['low_plus_25min'].shift(-1)
    df_joined['low_plus_35min'] = df_joined['low_plus_30min'].shift(-1)
    df_joined['low_plus_40min'] = df_joined['low_plus_35min'].shift(-1)
    df_joined['low_plus_45min'] = df_joined['low_plus_40min'].shift(-1)
    df_joined['low_plus_50min'] = df_joined['low_plus_45min'].shift(-1)
    df_joined['low_plus_55min'] = df_joined['low_plus_50min'].shift(-1)
    df_joined['low_plus_60min'] = df_joined['low_plus_55min'].shift(-1)
    
    df_joined['y_60min_max'] = df_joined[columns_for_y_60min_max].max(axis=1)
    df_joined['y_60min_min'] = df_joined[columns_for_y_60min_min].min(axis=1)
    
    df_joined['high_0min_60min_difference'] = df_joined['high'] - df_joined['high_plus_60min']
    df_joined['low_0min_60min_difference'] = df_joined['low'] - df_joined['low_plus_60min']
    
    df_joined['trend'] = df_joined.apply(define_the_trend, axis=1)
    
    one_hot_encoded = pd.get_dummies(df_joined['trend'], prefix='y_trend', dtype=int)
    
    df_joined = pd.concat([df_joined, one_hot_encoded], axis=1)
    
    
    df_joined.drop(['open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_bid', 'volume_ask', 'Gmt time_bid', 'Gmt time_ask'], inplace=True, axis=1)
    
    df_joined.drop(columns_for_y_60min_max[1:], inplace=True, axis=1)
    df_joined.drop(columns_for_y_60min_min[1:], inplace=True, axis=1)
    
    df_joined.drop(['high_0min_60min_difference'], inplace=True, axis=1)
    df_joined.drop(['low_0min_60min_difference'], inplace=True, axis=1)
    
    df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]
    
    df_joined_train = df_joined.iloc[0:712981]
    df_joined_test = df_joined.iloc[712981:950641]
    df_joined_val = df_joined.iloc[950641:]

    return df_joined_train, df_joined_test, df_joined_val