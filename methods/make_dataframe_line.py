import pandas as pd
import MetaTrader5 as mt
from methods.adx_calculation import calculate_adx
from methods.make_clean_dataframe_from_server import make_clean_dataframe_from_server
import numpy as np
import ta


def make_dataframe_line(timeframe, start_pos:int, end_pos:int, columns_order:list) -> pd.DataFrame:
    
   SYMBOL = [
   "EURUSD", # 0
   "AUDUSD", # 1
   "USDCAD", # 2
   "USDJPY", # 3
   "XBRUSD", # 4
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

   df_joined['40_min_vol'] = df_joined['close'].pct_change().rolling(8).std()
   df_joined['20_min_ma'] = df_joined['close'].rolling(4).mean()
   
   df_joined['day_of_week'] = df_joined.index.dayofweek
   df_joined['is_month_end'] = df_joined.index.is_month_end.astype(int)    
   
   df_joined['volume_ma_20_min'] = df_joined['volume'].rolling(4).mean()
   df_joined['volume_ma_ratio'] = df_joined['volume'] / df_joined['volume_ma_20_min']


   # print(df_joined.loc[df_joined.isna().any(axis=1)].head(50))

   df_joined = df_joined.loc[~df_joined.isna().any(axis=1)]
   
   df_joined.sort_index(axis=1, ascending=True, inplace=True)
   
   df_joined = df_joined.loc[:, columns_order]

   return df_joined