import pandas as pd

def make_minute_columns(df:pd.DataFrame) -> str:
    if(df['division_by_5_leftover'] == 0.0):
        return '1st_minute_volume'
    elif(df['division_by_5_leftover'] == 60.0):
        return '2nd_minute_volume'
    elif(df['division_by_5_leftover'] == 120.0):
        return '3rd_minute_volume'
    elif(df['division_by_5_leftover'] == 180.0):
        return '4th_minute_volume'
    elif(df['division_by_5_leftover'] == 240.0):
        return '5th_minute_volume'
    
    
def make_5min_volume_from_1min_volume(df_joined:pd.DataFrame) -> pd.DataFrame:

    df_joined.index = pd.to_datetime(df_joined.index, format='mixed')
    df_joined['index_dt'] = df_joined.index.map(pd.Timestamp.timestamp)
    df_joined['division_by_5_leftover'] = df_joined['index_dt'] % 300
    df_joined['timestamp_5min'] = df_joined['index_dt'] + (300-df_joined['division_by_5_leftover'])
    df_joined['timestamp_5min'] = pd.to_datetime(df_joined['timestamp_5min'], unit='s')
    df_joined['minute_columns'] = df_joined.apply(make_minute_columns, axis=1)
    df_joined.drop(columns=['open', 'close', 'high', 'low', 'index_dt', 'division_by_5_leftover'], inplace=True, axis=1)
    df_joined_pivot = pd.pivot_table(data=df_joined, values=['volume'], columns=['minute_columns'], index=['timestamp_5min'])
    df_joined_pivot = df_joined_pivot.droplevel(0,axis=1)
    df_joined_pivot = df_joined_pivot.reset_index()
    df_joined_pivot.set_index(keys=['timestamp_5min'], inplace=True)

    return df_joined_pivot
