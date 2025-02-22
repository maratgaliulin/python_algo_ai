import pandas as pd

def return_open_price(df:pd.DataFrame):
    if(df['open_bid'] > df['close_bid']):
        return df['open_bid']
    else:
        return df['open_ask']

def return_close_price(df:pd.DataFrame):
    if(df['open_bid'] > df['close_bid']):
        return df['close_ask']
    else:
        return df['close_bid']

def make_single_df_from_bid_ask(df_bid:pd.DataFrame, df_ask:pd.DataFrame) -> pd.DataFrame:

    df_joined = pd.merge(df_bid, df_ask, how='outer', left_index=True, right_index=True, suffixes=['_bid', '_ask'])
    df_joined['open'] = df_joined.apply(return_open_price, axis=1)
    df_joined['close'] = df_joined.apply(return_close_price, axis=1)
    df_joined['high'] = df_joined['high_ask']
    df_joined['low'] = df_joined['low_bid']
    df_joined['volume'] = (df_joined['volume_bid'] + df_joined['volume_ask']) / 2
    df_joined.drop(['open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask'], inplace=True, axis=1)
    
    return df_joined