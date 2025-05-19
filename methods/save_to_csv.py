import pandas as pd
from datetime import datetime

def save_to_csv(df_to_csv:pd.DataFrame, csv_address:str):
    """Метод сохраняет полученный с сервера датафрейм в файл csv. Перед сохранением строки в файл метод проверяет, есть ли данная строка в датафрейме. Если данной строки нет, происходит сохранение.

    Args:
        df_to_csv (pd.DataFrame): датафрейм, который нужно вписать в файл csv
        csv_address (str): адрес файла csv, в который нужно вписать новые данные
    """
    
    dt_pattern = '%Y-%m-%d %H:%M:%S'
    
    order_blocks_df = pd.read_csv(csv_address)
    # print(order_blocks_df)

    df_index_list = df_to_csv.index.strftime(dt_pattern).to_list()
    file_index_list = order_blocks_df['time'].to_list()

    # print('**********************')
    # print(type(df_index_list[0]))
    # print('**********************')
    # print(type(file_index_list[0]))
    
    df_columns = ['open', 'high', 'low', 'close']
    
    for df_idx in df_index_list:
        # print(df_idx)
        date_time_not_in_file = df_idx not in file_index_list
        if (date_time_not_in_file):
            df_idx_to_datetime = datetime.strptime(df_idx, dt_pattern)
            # print(df_idx_to_datetime)
            new_df = df_to_csv.loc[df_to_csv.index == df_idx_to_datetime].copy()
            # print(new_df)
            new_df.to_csv(csv_address, index=True, sep=',', header=None, mode='a', columns=df_columns)
    
    order_blocks_df = pd.read_csv(csv_address)

    order_blocks_df = order_blocks_df.loc[order_blocks_df['time'].str.len() == 19]

    order_blocks_df.to_csv(csv_address, index=False, sep=',', mode='w')

    return