import pandas as pd

def predicted_dataframe_indexing(predicted_dataframe: pd.DataFrame, dataframe_line:pd.DataFrame) -> pd.DataFrame:
    predicted_first_candle = predicted_dataframe.iloc[0]
    actual_last_candle = dataframe_line.iloc[-1]
    
    open_index = actual_last_candle['open'] / predicted_first_candle['open']
    high_index = actual_last_candle['high'] / predicted_first_candle['high']
    low_index = actual_last_candle['low'] / predicted_first_candle['low']
    close_index = actual_last_candle['close'] / predicted_first_candle['close']

    predicted_dataframe.loc[:, 'open'] *= open_index
    predicted_dataframe.loc[:,'high'] *= high_index
    predicted_dataframe.loc[:,'low'] *= low_index
    predicted_dataframe.loc[:,'close'] *= close_index

    print('indices:')
    print(open_index, high_index, low_index, close_index)

    return predicted_dataframe