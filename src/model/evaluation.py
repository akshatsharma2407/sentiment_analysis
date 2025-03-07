import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import pickle
import json
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import logging
import mlflow
import dagshub

# dagshub.init(repo_owner='akshatsharma2407', repo_name='sentiment_analysis', mlflow=True)


mlflow.set_tracking_uri('http://ec2-34-201-37-245.compute-1.amazonaws.com:5000/')


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

streamhandler = logging.StreamHandler()
streamhandler.setLevel('DEBUG')

filehandler = logging.FileHandler('errors.log')
filehandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamhandler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(streamhandler)
logger.addHandler(filehandler)

def load_model(model_path : str) -> BaseEstimator:
    try:
        clf = pickle.load(open(model_path,'rb'))
        logger.debug('model loaded !!')
        return clf
    except FileNotFoundError:
        logger.error('model.pkl not found')
        raise
    except Exception as e:
        logger.error('some error occured while loading the model', e)
        raise

def load_data(data_path : str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(data_path)
        logger.debug('data loaded !!')
        return test_data
    except FileNotFoundError:
        logger.error('failed to load data')
        raise
    except Exception as e:
        logger.error('some error occured while loading the data', e)
        raise

def evaluating(test_data : pd.DataFrame,clf : BaseEstimator) -> dict:
    try:
        X_test = test_data.iloc[:,0:-1].values
        y_test = test_data.iloc[:,-1].values

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }

        logger.debug('evaluation completed')

        return metrics_dict
    except Exception as e:
        logger.error('some error occured while evaluating the model ',e)
        raise

def export_metrics(metrics_dict : dict) -> dict:
    try:
        with open('reports/metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)
        logger.debug('evaluation metrics has been exported')
    except Exception as e:
        logger.error('some error occured while exporting the metrics', e)
        raise

def main():
    try:
        clf = load_model('models/model.pkl')
        test_data = load_data('./data/features/test_bow.csv')
        metrics_dict = evaluating(test_data,clf)
        export_metrics(metrics_dict)
        logger.debug('main function executed !!')

        mlflow.set_experiment('demo')
        with mlflow.start_run():
            mlflow.log_metrics(metrics_dict)
    except Exception as e:
        logger.error('some error occured while executing the main function' ,e )
        raise
mlflow.set_experiment('demo')
with mlflow.start_run():
    mlflow.log_artifact(__file__)
if __name__ == "__main__":
    main()