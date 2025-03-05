import numpy as np
import pandas as pd
import yaml
import os
from sklearn.feature_extraction.text import CountVectorizer
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

filehandler = logging.FileHandler('errors.log')
filehandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)

def load_params(path : str) -> int:
    try:
        max_features = yaml.safe_load(open(path,'r'))['features']['max_features']
        logger.debug('params loaded !!')
        return max_features
    except FileNotFoundError:
        logger.error('file not found')
        raise
    except Exception as e:
        logger.error('some error occured in load params ',e)
        raise


def load_data(train_data_path : str,test_data_path : str) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
    # fetch the data from data/processed
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        return train_data,test_data
    except FileNotFoundError : 
        logger.error('file not found')
        raise
    except Exception as e:
        logger.error('some error occured while loading data',e)
        raise

def splitting(train_data : pd.DataFrame,test_data : pd.DataFrame) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    try:
        train_data.fillna('',inplace=True)
        test_data.fillna('',inplace=True)

        # apply BoW
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        return X_train,y_train,X_test,y_test
    except Exception as e:
        logger.error('some error occured ', e)
        raise

def vectorizor(X_train : np.ndarray,X_test : np.ndarray,y_train : np.ndarray,y_test : np.ndarray,max_features : int) -> tuple[pd.DataFrame,pd.DataFrame]:

    try:

        # Apply Bag of Words (CountVectorizer)
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit the vectorizer on the training data and transform it
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform the test data using the same vectorizer
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())

        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())

        test_df['label'] = y_test

        logger.debug('vectorizor step completed')

        return train_df,test_df
    except Exception as e:
        logger.error('some error occured while vectorization')
        raise

def export(train_df : pd.DataFrame,test_df : pd.DataFrame) -> None:
    try:
        # store the data inside data/features
        data_path = os.path.join("data","features")

        os.makedirs(data_path)

        train_df.to_csv(os.path.join(data_path,"train_bow.csv"))
        test_df.to_csv(os.path.join(data_path,"test_bow.csv"))
        logger.debug('data exported successfully')
    except Exception as e:
        logger.critical('some error occured while exporting data')
        raise


def main() -> None:
    try:
        max_features = load_params('params.yaml')
        train_data,test_data = load_data('./data/processed/train_processed.csv','./data/processed/test_processed.csv')
        X_train,y_train,X_test,y_test = splitting(train_data,test_data)
        train_df,test_df = vectorizor(X_train,X_test,y_train,y_test,max_features)
        export(train_df,test_df)
        logger.debug('main function executed !!')
    except Exception as e:
        logger.critical('some error occured while executing main function')
        raise

if __name__ == '__main__':
    main()