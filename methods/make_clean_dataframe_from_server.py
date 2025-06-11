import pandas as pd
import MetaTrader5 as mt

def make_clean_dataframe_from_server(symbol:str, timeframe, start_pos:int, end_pos:int) -> pd.DataFrame:
   mt_dataframe_raw = mt.copy_rates_from_pos(symbol, timeframe, start_pos, end_pos)
   df = pd.DataFrame(mt_dataframe_raw)
   df['time']=pd.to_datetime(df['time'], unit='s')
   df.set_index(['time'], inplace=True)
   df.rename(columns={'tick_volume': 'volume'}, inplace=True)
   df.drop(['spread', 'real_volume'], axis=1, inplace=True) 
   df.sort_index(ascending=True, inplace=True)

   return df