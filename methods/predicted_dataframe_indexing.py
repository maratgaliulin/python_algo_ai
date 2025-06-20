import pandas as pd

def return_close_price(df:pd.DataFrame, border_price_high:float, border_price_low:float):
    if(df['open'] > df['close']):
        return df['close'] + border_price_low
    else:
        return df['close'] + border_price_high

def return_open_price(df:pd.DataFrame, border_price_high:float, border_price_low:float):
    if(df['open'] > df['close']):
        return df['close'] + border_price_high
    else:
        return df['close'] + border_price_low


def predicted_dataframe_indexing(predicted_dataframe: pd.DataFrame, dataframe_line:pd.DataFrame) -> pd.DataFrame:
    predicted_first_candle = predicted_dataframe.iloc[0]
    actual_last_candle = dataframe_line.iloc[-1]

    predicted_first_candle_is_red = predicted_first_candle['open'] > predicted_first_candle['close']
    predicted_first_candle_is_green = predicted_first_candle['open'] <= predicted_first_candle['close']

    actual_last_candle_is_red = actual_last_candle['open'] > actual_last_candle['close']
    actual_last_candle_is_green = actual_last_candle['open'] <= actual_last_candle['close']

    open_index = actual_last_candle['open'] / predicted_first_candle['open']
    high_index = actual_last_candle['high'] / predicted_first_candle['high']
    low_index = actual_last_candle['low'] / predicted_first_candle['low']
    close_index = actual_last_candle['close'] / predicted_first_candle['close']

    if(predicted_first_candle_is_red and actual_last_candle_is_red):
        border_price_high = 

    difference_index_high = actual_last_candle['high'] - predicted_first_candle['high'] 
    difference_index_low = actual_last_candle['low'] - predicted_first_candle['low']


    predicted_dataframe.loc[:, 'open'] *= open_index
    predicted_dataframe.loc[:,'high'] *= high_index
    predicted_dataframe.loc[:,'low'] *= low_index
    predicted_dataframe.loc[:,'close'] *= close_index

    print('indices:')
    print(open_index, high_index, low_index, close_index)

    return predicted_dataframe