import pandas as pd
import MetaTrader5 as mt
from methods.adx_calculation import calculate_adx
import numpy as np
import ta

def make_clean_dataframe_from_server(symbol:str, timeframe, start_pos:int, end_pos:int) -> pd.DataFrame:
   mt_dataframe_raw = mt.copy_rates_from_pos(symbol, timeframe, start_pos, end_pos)
   df = pd.DataFrame(mt_dataframe_raw)
   df['time']=pd.to_datetime(df['time'], unit='s')
   df.set_index(['time'], inplace=True)
   df.rename(columns={'tick_volume': 'volume'}, inplace=True)
   df.drop(['spread', 'real_volume'], axis=1, inplace=True)
   df.sort_index(ascending=True, inplace=True)

   return df

def make_dataframe_line(timeframe, start_pos:int, end_pos:int) -> pd.DataFrame:
    
   SYMBOL = [
   "EURUSD", # 0
   "AUDUSD", # 1
   "USDCAD", # 2
   "USDJPY", # 3
   "XBRUSD", # 4
   ]

   columns_order = [
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
        "open_jpyusd",
        "high_jpyusd",
        "low_jpyusd",
        "close_jpyusd",
        "volume_jpyusd"
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


   # print(df_joined.loc[df_joined.isna().any(axis=1)].head(50))



   df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]
   
   df_joined.sort_index(axis=1, ascending=True, inplace=True)
   
   df_joined = df_joined.loc[:, columns_order]

   return df_joined