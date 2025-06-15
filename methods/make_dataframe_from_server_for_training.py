import pandas as pd
import MetaTrader5 as mt
from methods.adx_calculation import calculate_adx
from methods.make_clean_dataframe_from_server import make_clean_dataframe_from_server
import numpy as np
import ta



def make_dataframe_from_server_for_training(timeframe, start_pos:int, end_pos:int, columns_order:list) -> pd.DataFrame:
    SYMBOL = [
    "EURUSD", # 0
    "AUDUSD", # 1
    "USDCAD", # 2
    "USDJPY", # 3
    "XBRUSD", # 4
    ]

    columns_for_reindex = columns_order.copy()    
    
    columns_for_y = [
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
    
    
    dataframe_eurusd = make_clean_dataframe_from_server(SYMBOL[0], timeframe, start_pos, end_pos)
    dataframe_audusd = make_clean_dataframe_from_server(SYMBOL[1], timeframe, start_pos, end_pos)
    dataframe_usdcad = make_clean_dataframe_from_server(SYMBOL[2], timeframe, start_pos, end_pos)
    dataframe_usdjpy = make_clean_dataframe_from_server(SYMBOL[3], timeframe, start_pos, end_pos)
    dataframe_xbrusd = make_clean_dataframe_from_server(SYMBOL[4], timeframe, start_pos, end_pos)

    df_joined = pd.merge(dataframe_eurusd, dataframe_audusd, how='inner', left_index=True, right_index=True, validate="one_to_one", suffixes=(None, '_audusd'))
    df_joined = pd.merge(df_joined, dataframe_xbrusd, how='inner', left_index=True, right_index=True, validate="one_to_one", suffixes=(None, '_brentusd'))
    df_joined = pd.merge(df_joined, dataframe_usdcad, how='inner', left_index=True, right_index=True, validate="one_to_one", suffixes=(None, '_cadusd'))
    df_joined = pd.merge(df_joined, dataframe_usdjpy, how='inner', left_index=True, right_index=True, validate="one_to_one", suffixes=(None, '_jpyusd'))

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

    df_joined['40_min_vol'] = df_joined['close'].pct_change().rolling(8).std()
    df_joined['20_min_ma'] = df_joined['close'].rolling(4).mean()
    
    df_joined['day_of_week'] = df_joined.index.dayofweek
    df_joined['is_month_end'] = df_joined.index.is_month_end.astype(int)    
    
    df_joined['volume_ma_20_min'] = df_joined['volume'].rolling(4).mean()
    df_joined['volume_ma_ratio'] = df_joined['volume'] / df_joined['volume_ma_20_min']

    
    df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]
    
    df_joined.sort_index(axis=1, ascending=True, inplace=True)

    for col in columns_for_y:
        columns_for_reindex.append(col)
    
    df_joined = df_joined.loc[:, columns_for_reindex]
    
    return df_joined