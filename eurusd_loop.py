from credentials import *
import time
import MetaTrader5 as mt
import pandas as pd
from methods.make_dataframe_line import make_dataframe_line

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

while True:
    mt_dataframe_raw = mt.copy_rates_from_pos("EURUSD", mt.TIMEFRAME_M5, 0, 13)
    # create DataFrame out of the obtained data
    dataframe_raw = pd.DataFrame(mt_dataframe_raw)
    # convert time in seconds into the datetime format
    dataframe_raw['time']=pd.to_datetime(dataframe_raw['time'], unit='s')
    dataframe_raw.set_index(['time'], inplace=True)
    dataframe_raw.sort_index(ascending=True, inplace=True)
    
    dataframe_line = make_dataframe_line(dataframe_raw)
    print(dataframe_line)


    
    time.sleep(5)