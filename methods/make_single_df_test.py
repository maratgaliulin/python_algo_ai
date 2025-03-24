import pandas as pd
from .return_single_large_dataframe import return_single_large_dataframe
from .open_close_high_low_prices import return_open_price, return_close_price, return_high_price, return_low_price
from .adx_calculation import calculate_adx
import numpy as np


def make_single_df_from_bid_test(df_bid:pd.DataFrame, df_ask:pd.DataFrame) -> pd.DataFrame:

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
    
    df_bid.rename(columns=rename_columns_dict, inplace=True)
    df_ask.rename(columns=rename_columns_dict, inplace=True)

    df_joined = pd.merge(df_bid, df_ask, how='outer', left_index=True, right_index=True, suffixes=['_bid', '_ask'])
    df_joined.fillna(inplace=True, value=values)
    df_joined['open'] = df_joined.apply(return_open_price, axis=1)
    
    df_joined['close'] = df_joined.apply(return_close_price, axis=1)
    
    df_joined['high'] = df_joined.apply(return_high_price, axis=1)
    
    df_joined['low'] = df_joined.apply(return_low_price, axis=1)
    
    df_joined['volume'] = df_joined['volume_bid'] + df_joined['volume_ask']
    
    
    df_joined.drop(['open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_bid', 'volume_ask'], inplace=True, axis=1)
    
    # print(df_joined_train.head())

    return df_joined