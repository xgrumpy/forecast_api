import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import optuna
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_data(datapath):
    return pd.read_csv(datapath)

def split_data(train_data):
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data['month'] = train_data['date'].dt.month
    train_data['day'] = train_data['date'].dt.dayofweek
    train_data['year'] = train_data['date'].dt.year
    col = ['store', 'item', 'month', 'day', 'year']
    y = 'sales'
    train_x, test_x, train_y, test_y = train_test_split(train_data[col], train_data[y], test_size=0.2, random_state=2018)
    return train_x, test_x, train_y, test_y, col

def train_model(train_x, train_y, test_x, test_y, col, use_tuning=False):
    if use_tuning:
        def objective(trial):
            params = {
                'nthread': 10,
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'boosting_type': 'gbdt',
                'objective': 'regression_l1',
                'metric': 'mape',
                'num_leaves': trial.suggest_int('num_leaves', 31, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0)
            }
            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_valid = lgb.Dataset(test_x, test_y)
            model = lgb.train(params, lgb_train, valid_sets=[lgb_valid], verbose_eval=False, early_stopping_rounds=50)
            preds = model.predict(test_x)
            mape = (abs(preds - test_y) / (abs(test_y) + 1)).mean()
            return mape

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params

        print(f"Best parameters: {best_params}")
        
        model = lgb.train(best_params, lgb.Dataset(train_x, train_y), 3000, valid_sets=[lgb.Dataset(test_x, test_y)], early_stopping_rounds=50, verbose_eval=50)
    else:
        params = {
            'nthread': 10,
            'max_depth': 5,
            'boosting_type': 'gbdt',
            'objective': 'regression_l1',
            'metric': 'mape',
            'num_leaves': 64,
            'learning_rate': 0.2,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 3.097758978478437,
            'lambda_l2': 2.9482537987198496,
            'min_child_weight': 6.996211413900573,
            'min_split_gain': 0.037310344962162616,
        }
        model = lgb.train(params, lgb.Dataset(train_x, train_y), 3000, valid_sets=[lgb.Dataset(test_x, test_y)], early_stopping_rounds=50, verbose_eval=50)
    
    joblib.dump(model, 'model/lgb_model.pkl')
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--tune', action='store_true', help='Use hyperparameter tuning')
    args = parser.parse_args()

    model_path = os.getenv('MODEL_PATH', 'model/lgb_model.pkl')
    data_path = os.getenv('DATA_PATH', 'data/train.csv')

    train_df = load_data(data_path)
    train_x, test_x, train_y, test_y, col = split_data(train_df)
    train_model(train_x, train_y, test_x, test_y, col, use_tuning=args.tune)
