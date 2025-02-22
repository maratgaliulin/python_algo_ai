import pandas as pd
import numpy as np
from methods.draw_graph import draw_static_graph
from methods.make_single_df_from_bid_ask import make_single_df_from_bid_ask

base_dir = "hist_data/EURUSD/5_min/"

dir_bid = base_dir + "Bid/EURUSD_5min_Bid_01.01.2004-01.01.2007.csv"
dir_ask = base_dir + "Ask/EURUSD_5min_Ask_01.01.2004-01.01.2007.csv"

rename_columns_dict = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}

df_5min_bid = pd.read_csv(dir_bid, index_col="Gmt time").sort_index(ascending=True)
df_5min_ask = pd.read_csv(dir_ask, index_col="Gmt time").sort_index(ascending=True)
df_5min_bid.rename(columns=rename_columns_dict, inplace=True)
df_5min_ask.rename(columns=rename_columns_dict, inplace=True)


df_5min_joined = make_single_df_from_bid_ask(df_5min_bid, df_5min_ask)

# print(df_5min_bid.tail(50))
# print('********************')
# print(df_5min_ask.tail(50))
# print('********************')
for i in range(0,90):
    print(df_5min_joined.iloc[i].T)

# draw_static_graph(df_5min_bid.tail(70), df_5min_ask.tail(70), df_5min_joined.tail(70))