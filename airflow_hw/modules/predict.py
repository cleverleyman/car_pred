import dill
import json
import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline

path = os.environ.get('PROJECT_PATH', '.')

def get_model() -> Pipeline:
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{latest_model}', 'rb') as file:
        model = dill.load(file)
    return model


def get_predict() -> pd.DataFrame:
    preds = {
        'car_id': [],
        'pred': [],
    }
    test_cars = os.listdir(f'{path}/data/test')
    model = get_model()

    for car_id in test_cars:
        with open(f'{path}/data/test/{car_id}', 'rb') as file:
            car = json.load(file)

        df = pd.DataFrame(car, index=[0])
        y = model.predict(df)
        preds['car_id'].append(car_id.split('.')[0])
        preds['pred'].append(y[0])
    return pd.DataFrame(preds)

def predict() -> None:
    predictions = get_predict()
    preds_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predictions.to_csv(preds_filename, index=False)
    logging.info(f'Predictions are saved as {preds_filename}')

if __name__ == '__main__':
    predict()
