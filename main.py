import pandas as pd
from methods.draw_graph import draw_static_graph

base_dir = "hist_data/EURUSD/5_min/"

dir_bid = base_dir + "Bid/EURUSD_5min_Bid_01.01.2004-01.01.2007.csv"
dir_ask = base_dir + "Ask/EURUSD_5min_Ask_01.01.2004-01.01.2007.csv"

df_5min_bid = pd.read_csv(dir_bid, index_col="Gmt time").sort_index(ascending=True)
df_5min_ask = pd.read_csv(dir_ask, index_col="Gmt time").sort_index(ascending=True)

print(df_5min_bid.head())

draw_static_graph(df_5min_bid.head(20), df_5min_ask.head(20))