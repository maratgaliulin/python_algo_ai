import pandas as pd
import os

def return_single_large_dataframe(base_dir:str, time_series_folder:str, Bid_or_Ask_folder:str) -> pd.DataFrame:
    total_dir = base_dir + time_series_folder + Bid_or_Ask_folder
    columns = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']
    final_df = pd.DataFrame(columns=columns)
    for file in os.listdir(total_dir):
        print(file)

    print(final_df)