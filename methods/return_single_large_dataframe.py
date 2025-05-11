import pandas as pd
import os

def return_single_large_dataframe(base_dir:str, time_series_folder:str, Bid_or_Ask_folder:str) -> pd.DataFrame:
    total_dir = base_dir + time_series_folder + Bid_or_Ask_folder
    columns = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']
    final_df = pd.DataFrame(columns=columns)
        
    for file in os.listdir(total_dir):
        final_dir = total_dir + file
        # new_df = pd.read_csv(final_dir, index_col="Gmt time", parse_dates=["Gmt time"], dayfirst=True).sort_index(ascending=True)
        new_df = pd.read_csv(final_dir)
        new_df["Gmt time"] = pd.to_datetime(new_df["Gmt time"], format='mixed', dayfirst=True)
        new_df.set_index(["Gmt time"], inplace=True)
        final_df = pd.concat([final_df, new_df])

    return final_df