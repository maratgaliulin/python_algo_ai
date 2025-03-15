import pandas as pd
from .return_single_large_dataframe import return_single_large_dataframe

def return_open_price(df:pd.DataFrame):
    if((df['open_ask'] != 999) and (df['open_bid'] != 999)):
        if((df['open_bid'] > df['close_bid']) or ((df['open_bid'] == df['close_bid']) and (df['open_ask'] > df['open_bid']))):        
            return df['open_ask']
        else:
            return df['open_bid']
            
    elif((df['open_bid'] != 999) and (df['open_ask'] == 999)):
        return df['open_bid']
    else:
        return df['open_ask']

def return_close_price(df:pd.DataFrame):
    if((df['close_bid'] != 999) and (df['close_ask'] != 999)):
        if((df['open_bid'] > df['close_bid']) or ((df['open_bid'] == df['close_bid']) and (df['open_ask'] > df['open_bid']))):        
            return df['close_bid']
        else:
            return df['close_ask']
    
    elif((df['close_ask'] != 999) and (df['close_bid'] == 999)):
        return df['close_ask']
    else:
        return df['close_bid']
    
def return_high_price(df:pd.DataFrame):
    if ((df['high_bid'] != 999) and (df['high_ask'] != 999)):
        if((df['high_bid'] > df['high_ask'])):
            return df['high_bid']
        else:
            return df['high_ask']
    
    elif((df['high_ask'] != 999) and (df['high_bid'] == 999)):
        return df['high_ask']
    else:
        return df['high_bid']
    
def return_low_price(df:pd.DataFrame):
    if((df['low_bid'] != 999) and (df['low_ask'] != 999)):
        if((df['low_bid'] < df['low_ask'])):
            return df['low_bid']
        else:
            return df['low_ask']
    
    elif((df['low_ask'] != 999) and (df['low_bid'] == 999)):
        return df['low_ask']
    else:
        return df['low_bid']

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

    df_bid = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_bid)
    df_ask = return_single_large_dataframe(base_dir=base_dir, time_series_folder=time_series_folder, Bid_or_Ask_folder=bid_or_ask_folder_ask)
    
    df_bid.rename(columns=rename_columns_dict, inplace=True)
    df_ask.rename(columns=rename_columns_dict, inplace=True)

    df_joined = pd.merge(df_bid, df_ask, how='outer', left_index=True, right_index=True, suffixes=['_bid', '_ask'])
    df_joined.fillna(inplace=True, value=values)
    df_joined['open'] = df_joined.apply(return_open_price, axis=1)
    df_joined['close'] = df_joined.apply(return_close_price, axis=1)
    df_joined['high'] = df_joined.apply(return_high_price, axis=1)
    df_joined['low'] = df_joined.apply(return_low_price, axis=1)
    df_joined['volume'] = df_joined['volume_bid'] + df_joined['volume_ask']
    df_joined.drop(['open_bid', 'high_bid', 'low_bid', 'close_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_bid', 'volume_ask', 'Gmt time_bid', 'Gmt time_ask'], inplace=True, axis=1)
    
    df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]

    return df_joined