import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

def search_optimal_parameters_for_random_forest_max_value_prediction(train_df:pd.DataFrame) -> None:
    param_grid = {'n_estimators': [200, 300, 700], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [1, 2, 3]}

    X_train = train_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = train_df['y_60min_max']
    
    logreg_model = RandomForestRegressor(random_state=1, verbose=1)

    grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy', verbose=1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best Parameters: {best_params}')
    print(f'Best Score: {best_score}')

    return

def predict_max_value_with_random_forest_regressor(train_df:pd.DataFrame, test_df:pd.DataFrame, validation_df:pd.DataFrame, base_dir:str):
    X_train = train_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = train_df['y_60min_max']
    y_test = test_df['y_60min_max']
    y_validation = validation_df['y_60min_max']
    
    
    logreg_model = RandomForestRegressor(random_state=1, n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=3, max_features='sqrt', max_samples=0.8, verbose=2)

    logreg_model.fit(X_train, y_train)

    with open(base_dir + '/random_forest_regressor_predict_max_values_before_test_training.pkl', 'wb') as file:
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

    with open(base_dir + '/random_forest_regressor_predict_max_values_after_test_training.pkl', 'wb') as file:
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
