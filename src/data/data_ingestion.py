import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(path : str) -> float:
    try:
        test_size = yaml.safe_load(open(path,'r'))['data_ingestion']['test_size']
        logger.debug('params loaded!')
        return test_size
    except FileNotFoundError:
        logger.error('file not found in load_params')
        raise
    except:
        logger.error('some error occured in load params')
        raise

def load_data(path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.debug('data loaded successfully')
        return df
    except FileNotFoundError:
        logger.error('file not found while loading the data')
        raise
    except Exception as e:
        logger.error('some error occured', e)
        raise

def cleanAndSplit(df : pd.DataFrame,test_size : float) -> tuple[pd.DataFrame,pd.DataFrame] :
    try:
        df.drop(columns=['tweet_id'],inplace=True)

        final_df = df[df['sentiment'].isin(['neutral','sadness'])]

        final_df['sentiment'].replace({'neutral':1, 'sadness':0},inplace=True)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        logger.debug('data splited')

        return train_data,test_data
    
    except Exception as e:
        logger.error('some error occured ', e)
        raise

def export(train_data : pd.DataFrame,test_data : pd.DataFrame) -> None:
    try:
        data_path = os.path.join("data","raw")

        os.makedirs(data_path)

        train_data.to_csv(os.path.join(data_path,"train.csv"))
        test_data.to_csv(os.path.join(data_path,"test.csv"))
        logger.info('raw data exported')

    except Exception as e:
        logger.critical('some error occured ',e)
        raise

def main() -> None:
    try:
        test_size = load_params('params.yaml')
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        train_data,test_data = cleanAndSplit(df,test_size)
        export(train_data,test_data)
        logger.info('data ingestion occured')

    except Exception as e:
        logger.error('some issue occured in main function ',e)
        raise

if __name__ == '__main__':
    main()