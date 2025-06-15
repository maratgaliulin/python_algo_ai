import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def generate_automatic_features_for_model_training(df_raw:pd.DataFrame, cols_order:list, column_for_y:str, base_dir:str) -> pd.DataFrame:

    columns_order = cols_order.copy()
    df_data = df_raw.loc[:, columns_order]
    df_data[column_for_y] = df_raw[column_for_y]
    df_data.reset_index(inplace=True)
    df_data.dropna(axis=1, inplace=True)
    df_data['id'] = df_data.index
    columns_order.append('id')
    columns_order.append('time')

    extracted_features = extract_features(
        df_data.loc[:, columns_order],
        column_id="id",  
        column_sort="time", 
        default_fc_parameters=EfficientFCParameters(),          
        impute_function=impute, 
    )

    y = df_data[column_for_y] 

    features_filtered = select_features(extracted_features, y)

    feature_columns = features_filtered.columns.to_list()

    def select_columns(X, columns=feature_columns):
        return X[columns]

    features_filtered['time'] = df_raw.index

    features_filtered.set_index(['time'], inplace=True, drop=True)

    print(f"Original number of features: {len(extracted_features.columns)}")
    print(f"Number of features after selection: {len(features_filtered.columns)}")
    print("Some selected features:")
    print(features_filtered.head())

    pipeline = Pipeline([
        ('imputer', FunctionTransformer(impute)),
        ('selector', FunctionTransformer(select_columns))
    ])

    joblib.dump(pipeline, f'{base_dir}/pipelines/feature_pipeline_{column_for_y}.pkl')
    joblib.dump(feature_columns, f'{base_dir}/feature_columns/feature_columns_{column_for_y}.pkl')

    return features_filtered


def generate_automatic_features_for_model_test(df_raw:pd.DataFrame, cols_order:list, base_dir:str, column_for_y:str) -> pd.DataFrame:

    columns_order = cols_order.copy()
    df_data = df_raw.loc[:, columns_order]
    df_data.reset_index(inplace=True)
    df_data.dropna(axis=1, inplace=True)
    df_data['id'] = df_data.index

    if('id' not in columns_order):
        columns_order.append('id')

    if('time' not in columns_order):
        columns_order.append('time')

    extracted_features = extract_features(
        df_data.loc[:, columns_order],
        column_id="id",  
        column_sort="time", 
        default_fc_parameters=EfficientFCParameters(),          
        impute_function=impute, 
    )

    pipeline = joblib.load(f'{base_dir}/pipelines/feature_pipeline_{column_for_y}.pkl')
    feature_columns = joblib.load(f'{base_dir}/feature_columns/feature_columns_{column_for_y}.pkl')

    production_ready_features = pipeline.transform(extracted_features)

    production_ready_features['time'] = df_raw.index

    production_ready_features.set_index(['time'], inplace=True, drop=True)

    print(f"Original number of features: {len(extracted_features.columns)}")
    print(f"Number of features after selection: {len(production_ready_features.columns)}")
    print("Some selected features:")
    print(production_ready_features.head())

    return production_ready_features.loc[:, feature_columns]