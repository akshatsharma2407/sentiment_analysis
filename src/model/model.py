import mlflow.models
import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
import os 
import logging
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV

# dagshub.init(repo_owner='akshatsharma2407', repo_name='sentiment_analysis', mlflow=True)


mlflow.set_tracking_uri('http://ec2-34-201-37-245.compute-1.amazonaws.com:5000/')

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

def training(train_data : pd.DataFrame) -> BaseEstimator:
    try:
        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values

        # Define and train the model

        clf = GradientBoostingClassifier()

        param_grid = {
            'n_estimators' : [10,29,59],
            'learning_rate' : [0.1,0.01,0.001]
        }

        grid_search = GridSearchCV(estimator=clf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)

        grid_search.fit(X_train, y_train)

        signature = mlflow.models.infer_signature(X_train,grid_search.best_estimator_.predict(X_train))
        mlflow.sklearn.log_model(grid_search.best_estimator_,'grandientboosting',signature=signature)

        mlflow.sklearn.log_model(clf,'gradient descent best model',signature=signature)

        logger.debug('model trained !!')
        return grid_search.best_estimator_
    
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
        mlflow.autolog()
        mlflow.set_experiment('demo')
        with mlflow.start_run():
            # estimators = load_params('params.yaml')
            train_data = load_data('./data/features/train_bow.csv')
            clf = training(train_data)

            dump_model(clf)
            logger.debug('main function executed')

    except Exception as e:
        logger.error('main function exeuted ', e)
        raise
    

if __name__ == '__main__':
    main()