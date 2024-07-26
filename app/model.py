import joblib
import pandas as pd

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_sales(date, store, item, model):
    data = {
        'date': [date],
        'store': [store],
        'item': [item]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    col = ['store', 'item', 'month', 'day', 'year']
    return model.predict(df[col])
