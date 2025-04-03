import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
import pandas as pd

def search_optimal_parameters_for_random_forest_trend_prediction(train_df:pd.DataFrame) -> None:
    param_grid = {'n_estimators': [200, 300, 700], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [1, 2, 3]}

    X_train = train_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = train_df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
    
    logreg_model = RandomForestClassifier(random_state=1, verbose=1)

    grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy', verbose=1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best Parameters: {best_params}')
    print(f'Best Score: {best_score}')

    return

def predict_trend_direction_with_random_forest_classifier(df:pd.DataFrame, test_df:pd.DataFrame, validation_df:pd.DataFrame, base_dir:str):
    
    columns_order = ['open', 'open_minus_5min', 'open_minus_10min', 'open_minus_15min',
       'open_minus_20min', 'open_minus_25min', 'open_minus_30min',
       'open_minus_35min', 'open_minus_40min', 'open_minus_45min',
       'open_minus_50min', 'open_minus_55min', 'open_minus_60min', 'close',
       'close_minus_5min', 'close_minus_10min', 'close_minus_15min',
       'close_minus_20min', 'close_minus_25min', 'close_minus_30min',
       'close_minus_35min', 'close_minus_40min', 'close_minus_45min',
       'close_minus_50min', 'close_minus_55min', 'close_minus_60min', 'high',
       'high_minus_5min', 'high_minus_10min', 'high_minus_15min',
       'high_minus_20min', 'high_minus_25min', 'high_minus_30min',
       'high_minus_35min', 'high_minus_40min', 'high_minus_45min',
       'high_minus_50min', 'high_minus_55min', 'high_minus_60min', 'low',
       'low_minus_5min', 'low_minus_10min', 'low_minus_15min',
       'low_minus_20min', 'low_minus_25min', 'low_minus_30min',
       'low_minus_35min', 'low_minus_40min', 'low_minus_45min',
       'low_minus_50min', 'low_minus_55min', 'low_minus_60min', 'volume',
       'volume_minus_5min', 'volume_minus_10min', 'volume_minus_15min',
       'volume_minus_20min', 'volume_minus_25min', 'volume_minus_30min',
       'volume_minus_35min', 'volume_minus_40min', 'volume_minus_45min',
       'volume_minus_50min', 'volume_minus_55min', 'volume_minus_60min',
       'open_normalized', 'close_normalized', 'high_normalized',
       'low_normalized', 'volume_normalized', 'open_log', 'close_log',
       'high_log', 'low_log', '+DI', '-DI', 'ADX']
    
    X_train = df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_train = X_train.reindex(columns=columns_order)
    X_test = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test = X_test.reindex(columns=columns_order)
    X_validation = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation = X_validation.reindex(columns=columns_order)

    y_train = df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
    y_test = test_df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
    y_validation = validation_df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
     
    logreg_model = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5, min_samples_split=10, min_samples_leaf=3, n_jobs=10, verbose=2)

    logreg_model.fit(X_train, y_train)

    with open(base_dir + '/random_forest_regressor_predict_trend_direction-2025-03-25.pkl', 'wb') as file:
        pickle.dump(logreg_model, file)

    print('*******************************')
    print('Before training on test sample:')
    print('*******************************')

    # print('Feature Importances:')
    # print(logreg_model.feature_importances_)

    print('Score on train sample:')
    print(logreg_model.score(X_train, y_train))

    print('Score on test sample:')
    print(logreg_model.score(X_test, y_test))

    print('Score on validation sample:')
    print(logreg_model.score(X_validation, y_validation))

    logreg_model.fit(X_test, y_test)

    with open(base_dir + '/random_forest_regressor_predict_trend_direction_after_test_training-2025-03-25.pkl', 'wb') as file:
        pickle.dump(logreg_model, file)

    print('*******************************')
    print('After training on test sample:')
    print('*******************************')

    # print('Feature Importances:')
    # print(logreg_model.feature_importances_)

    print('Score on train sample:')
    print(logreg_model.score(X_train, y_train))

    print('Score on test sample:')
    print(logreg_model.score(X_test, y_test))

    print('Score on validation sample:')
    print(logreg_model.score(X_validation, y_validation))
