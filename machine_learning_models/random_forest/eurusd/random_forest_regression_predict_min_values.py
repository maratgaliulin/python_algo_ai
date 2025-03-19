import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd

def search_optimal_parameters_for_random_forest_min_value_prediction(train_df:pd.DataFrame) -> None:
    param_grid = {'n_estimators': [200, 300, 700], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [1, 2, 3]}

    X_train = train_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = train_df['y_60min_min']
    
    logreg_model = RandomForestRegressor(random_state=1, verbose=2)

    grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy', verbose=2)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best Parameters: {best_params}')
    print(f'Best Score: {best_score}')

    return

def predict_min_value_with_random_forest_regressor(train_df:pd.DataFrame, test_df:pd.DataFrame, validation_df:pd.DataFrame):
    X_train = train_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = train_df['y_60min_min']
    y_test = test_df['y_60min_min']
    y_validation = validation_df['y_60min_min']
    
    
    logreg_model = RandomForestRegressor(random_state=1, n_estimators=2000, max_depth=2, min_samples_split=3, verbose=1)

    logreg_model.fit(X_train, y_train)

    with open('machine_learning_models/eurusd/pickle_files/random_forest_regressor_predict_min_values__before_test_training.pkl', 'wb') as file:
        pickle.dump(logreg_model, file)

    print('*******************************')
    print('Before training on test sample:')
    print('*******************************')

    print('Feature Importances:')
    print(logreg_model.feature_importances_)

    print('Score on train sample:')
    logreg_model.score(X_train, y_train)

    print('Score on test sample:')
    logreg_model.score(X_test, y_test)

    print('Score on validation sample:')
    logreg_model.score(X_validation, y_validation)

    logreg_model.fit(X_test, y_test)

    with open('machine_learning_models/eurusd/pickle_files/random_forest_regressor_predict_min_values_after_test_training.pkl', 'wb') as file:
        pickle.dump(logreg_model, file)

    print('*******************************')
    print('After training on test sample:')
    print('*******************************')

    print('Feature Importances:')
    print(logreg_model.feature_importances_)

    print('Score on train sample:')
    logreg_model.score(X_train, y_train)

    print('Score on test sample:')
    logreg_model.score(X_test, y_test)

    print('Score on validation sample:')
    logreg_model.score(X_validation, y_validation)
