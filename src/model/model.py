import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
import os 
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(path : str) -> int:
    try:
        estimators = yaml.safe_load(open(path,'r'))['model']['n_estimators']
        return estimators
    except FileNotFoundError :
        logger.error('params.yaml file not found')
        raise
    except Exception as e:
        logger.error('some error occured while loading the params')
        raise

def load_data(path : str) -> pd.DataFrame:
    try:
        # fetch the data from data/processed
        train_data = pd.read_csv(path)
        return train_data
    except FileNotFoundError :
        logger.error('data file not found')
        raise
    except Exception as e:
        logger.error('some error occured while loading data')
        raise

def training(train_data : pd.DataFrame,estimators : int) -> BaseEstimator:
    try:
        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values

        # Define and train the model

        clf = GradientBoostingClassifier(n_estimators=estimators)
        clf.fit(X_train, y_train)
        logger.debug('model trained !!')
        return clf
    
    except Exception as e:
        logger.error('some error occured while training the model ',e)
        raise

def dump_model(clf : BaseEstimator) -> None:
    try:
        # save
        pickle.dump(clf, open('models/model.pkl','wb'))
        logger.debug('model dumped !!')
    except Exception as e:
        logger.error('failed to dump the model ',e)
        raise

def main() -> None:
    try:
        estimators = load_params('params.yaml')
        train_data = load_data('./data/features/train_bow.csv')
        clf = training(train_data,estimators)
        dump_model(clf)
        logger.debug('main function executed')

    except Exception as e:
        logger.error('main function exeuted ', e)
        raise

if __name__ == '__main__':
    main()