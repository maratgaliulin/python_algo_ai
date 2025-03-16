import pandas as pd
from methods.draw_graph import draw_static_graph
from methods.make_single_df_from_bid_ask import make_single_df_from_bid_ask


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



base_dir = "hist_data/EURUSD/5_min/"

dir_bid = base_dir + "Bid/EURUSD_5min_Bid_01.01.2004-01.01.2007.csv"
dir_ask = base_dir + "Ask/EURUSD_5min_Ask_01.01.2004-01.01.2007.csv"


# df_5min_bid = pd.read_csv(dir_bid, index_col="Gmt time").sort_index(ascending=True)
# df_5min_ask = pd.read_csv(dir_ask, index_col="Gmt time").sort_index(ascending=True)





# print(df_5min_bid.tail(50))
# print('********************')
# print(df_5min_ask.tail(50))
# print('********************')
# for i in range(0,90):
#     print(df_5min_joined.iloc[i].T)

# draw_static_graph(df_5min_bid.tail(70), df_5min_ask.tail(70), df_5min_joined.tail(70))

bdir = "hist_data/"
time_series_folder = "EURUSD/5_min/"
bid_or_ask_folder_bid = "Bid/"
bid_or_ask_folder_ask = "Ask/"




df_5min_joined = make_single_df_from_bid_ask(
    base_dir=bdir, 
    time_series_folder=time_series_folder, 
    bid_or_ask_folder_bid=bid_or_ask_folder_bid, 
    bid_or_ask_folder_ask=bid_or_ask_folder_ask
    )

print(df_5min_joined.head(20))
print(df_5min_joined.tail(20))
print(len(df_5min_joined))
# print()