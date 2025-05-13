import pandas as pd
import os

def return_single_large_dataframe(base_dir:str, time_series_folder:str, Bid_or_Ask_folder:str) -> pd.DataFrame:
    total_dir = base_dir + time_series_folder + Bid_or_Ask_folder
    # columns = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']
    rename_columns_dict = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
    }

    dataframe_file_name = total_dir + os.listdir(total_dir)[0]
    new_df = pd.read_csv(dataframe_file_name, parse_dates=["Gmt time"], dayfirst=True)
    # new_df["Gmt time"] = pd.to_datetime(new_df["Gmt time"], dayfirst=True)
    new_df.set_index(["Gmt time"], drop=True, inplace=True)

    new_df.rename(columns=rename_columns_dict, inplace=True)
    new_df.sort_index(ascending=True, inplace=True)

    # print(os.listdir(total_dir)[0], ':\n\n')
    # print(new_df.describe())
    # print(new_df.info())

    return new_df