import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
import pandas as pd

def search_optimal_parameters_for_gradient_boost_trend_prediction(train_df:pd.DataFrame) -> None:
    param_grid = {'n_estimators': [200, 300, 700], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [1, 2, 3]}

    X_train = train_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = train_df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
    
    logreg_model = GradientBoostingClassifier(random_state=1, verbose=1)

    grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy', verbose=1)

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best Parameters: {best_params}')
    print(f'Best Score: {best_score}')

    return

def predict_trend_direction_with_gradient_boost_classifier(df:pd.DataFrame, test_df:pd.DataFrame, validation_df:pd.DataFrame, base_dir: str):
    X_train = df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_test = test_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])
    X_validation = validation_df.drop(columns=['y_60min_max', 'y_60min_min', 'trend', 'y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'])

    y_train = df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
    y_test = test_df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]
    y_validation = validation_df[['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined']]

    y_train['trend'] = y_train.idxmax(axis=1)
    y_train['trend'] = y_train['trend'].str.replace('y_trend_', '')
    y_train.drop(columns=['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'], axis=1, inplace=True)
    y_train = y_train['trend']
    
    y_test['trend'] = y_test.idxmax(axis=1)
    y_test['trend'] = y_test['trend'].str.replace('y_trend_', '')
    y_test.drop(columns=['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'], axis=1, inplace=True)
    y_test = y_test['trend']

    y_validation['trend'] = y_validation.idxmax(axis=1)
    y_validation['trend'] = y_validation['trend'].str.replace('y_trend_', '')
    y_validation.drop(columns=['y_trend_downtrend', 'y_trend_uptrend', 'y_trend_trend undefined'], axis=1, inplace=True)
    y_validation = y_validation['trend']

     
    logreg_model = GradientBoostingClassifier(random_state=1, n_estimators=500, max_depth=4, learning_rate=0.01, min_samples_leaf=3, verbose=2)

    logreg_model.fit(X_train, y_train)

    with open(base_dir + '/xgboost_regressor_predict_trend_direction-2025-03-27.pkl', 'wb') as file:
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

    with open(base_dir + '/xgboost_regressor_predict_trend_direction_after_test_training-2025-03-27.pkl', 'wb') as file:
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
