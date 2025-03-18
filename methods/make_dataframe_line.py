import pandas as pd
from methods.adx_calculation import calculate_adx
import numpy as np

def make_dataframe_line(df:pd.DataFrame) -> pd.DataFrame:

    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df.drop(['spread', 'real_volume'], axis=1, inplace=True)
    
    df['open_minus_5min'] = df['open'].shift(1)
    df['open_minus_10min'] = df['open_minus_5min'].shift(1)
    df['open_minus_15min'] = df['open_minus_10min'].shift(1)
    df['open_minus_20min'] = df['open_minus_15min'].shift(1)
    df['open_minus_25min'] = df['open_minus_20min'].shift(1)
    df['open_minus_30min'] = df['open_minus_25min'].shift(1)
    df['open_minus_35min'] = df['open_minus_30min'].shift(1)
    df['open_minus_40min'] = df['open_minus_35min'].shift(1)
    df['open_minus_45min'] = df['open_minus_40min'].shift(1)
    df['open_minus_50min'] = df['open_minus_45min'].shift(1)
    df['open_minus_55min'] = df['open_minus_50min'].shift(1)
    df['open_minus_60min'] = df['open_minus_55min'].shift(1)

    df['close_minus_5min'] = df['close'].shift(1)
    df['close_minus_10min'] = df['close_minus_5min'].shift(1)
    df['close_minus_15min'] = df['close_minus_10min'].shift(1)
    df['close_minus_20min'] = df['close_minus_15min'].shift(1)
    df['close_minus_25min'] = df['close_minus_20min'].shift(1)
    df['close_minus_30min'] = df['close_minus_25min'].shift(1)
    df['close_minus_35min'] = df['close_minus_30min'].shift(1)
    df['close_minus_40min'] = df['close_minus_35min'].shift(1)
    df['close_minus_45min'] = df['close_minus_40min'].shift(1)
    df['close_minus_50min'] = df['close_minus_45min'].shift(1)
    df['close_minus_55min'] = df['close_minus_50min'].shift(1)
    df['close_minus_60min'] = df['close_minus_55min'].shift(1)

    df['high_minus_5min'] = df['high'].shift(1)
    df['high_minus_10min'] = df['high_minus_5min'].shift(1)
    df['high_minus_15min'] = df['high_minus_10min'].shift(1)
    df['high_minus_20min'] = df['high_minus_15min'].shift(1)
    df['high_minus_25min'] = df['high_minus_20min'].shift(1)
    df['high_minus_30min'] = df['high_minus_25min'].shift(1)
    df['high_minus_35min'] = df['high_minus_30min'].shift(1)
    df['high_minus_40min'] = df['high_minus_35min'].shift(1)
    df['high_minus_45min'] = df['high_minus_40min'].shift(1)
    df['high_minus_50min'] = df['high_minus_45min'].shift(1)
    df['high_minus_55min'] = df['high_minus_50min'].shift(1)
    df['high_minus_60min'] = df['high_minus_55min'].shift(1)

    df['low_minus_5min'] = df['low'].shift(1)
    df['low_minus_10min'] = df['low_minus_5min'].shift(1)
    df['low_minus_15min'] = df['low_minus_10min'].shift(1)
    df['low_minus_20min'] = df['low_minus_15min'].shift(1)
    df['low_minus_25min'] = df['low_minus_20min'].shift(1)
    df['low_minus_30min'] = df['low_minus_25min'].shift(1)
    df['low_minus_35min'] = df['low_minus_30min'].shift(1)
    df['low_minus_40min'] = df['low_minus_35min'].shift(1)
    df['low_minus_45min'] = df['low_minus_40min'].shift(1)
    df['low_minus_50min'] = df['low_minus_45min'].shift(1)
    df['low_minus_55min'] = df['low_minus_50min'].shift(1)
    df['low_minus_60min'] = df['low_minus_55min'].shift(1)

    df['volume_minus_5min'] = df['volume'].shift(1)
    df['volume_minus_10min'] = df['volume_minus_5min'].shift(1)
    df['volume_minus_15min'] = df['volume_minus_10min'].shift(1)
    df['volume_minus_20min'] = df['volume_minus_15min'].shift(1)
    df['volume_minus_25min'] = df['volume_minus_20min'].shift(1)
    df['volume_minus_30min'] = df['volume_minus_25min'].shift(1)
    df['volume_minus_35min'] = df['volume_minus_30min'].shift(1)
    df['volume_minus_40min'] = df['volume_minus_35min'].shift(1)
    df['volume_minus_45min'] = df['volume_minus_40min'].shift(1)
    df['volume_minus_50min'] = df['volume_minus_45min'].shift(1)
    df['volume_minus_55min'] = df['volume_minus_50min'].shift(1)
    df['volume_minus_60min'] = df['volume_minus_55min'].shift(1)

    df_open_mean = df['open'].mean()
    df_close_mean = df['close'].mean()
    df_high_mean = df['high'].mean()
    df_low_mean = df['low'].mean()
    df_volume_mean = df['volume'].mean()
    
    df_open_std = df['open'].std()
    df_close_std = df['close'].std()
    df_high_std = df['high'].std()
    df_low_std = df['low'].std()
    df_volume_std = df['volume'].std()
    
    df['open_normalized'] = (df['open'] - df_open_mean)/df_open_std
    df['close_normalized'] = (df['close'] - df_close_mean)/df_close_std
    df['high_normalized'] = (df['high'] - df_high_mean)/df_high_std
    df['low_normalized'] = (df['low'] - df_low_mean)/df_low_std
    df['volume_normalized'] = (df['volume'] - df_volume_mean)/df_volume_std
    
    df['open_log'] = np.log(df['open'])
    df['close_log'] = np.log(df['close'])
    df['high_log'] = np.log(df['high'])
    df['low_log'] = np.log(df['low'])
    
    df = calculate_adx(df, period=12)

    df = df.loc[~df.isna().any(axis=1)]

    return df