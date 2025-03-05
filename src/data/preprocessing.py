import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(train_path : str,test_path : str) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        # fetch the data from data/raw
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('data loaded  !')
        return train_data,test_data
    except FileNotFoundError:
        logger.error('data loaded !!')
        raise
    except Exception as e:
        logger.error('some error occured in load data', e)
        raise

def downloading_nltk_dependencies() -> None:
    try:
        # transform the data
        nltk.download('wordnet')
        nltk.download('stopwords')
        logger.debug('nltk utility downloaded')
    except Exception as e:
        logger.error('some error occured while downloading nltk utility', e)
        raise

def lemmatization(text):
    try:
        lemmatizer= WordNetLemmatizer()

        text = text.split()

        text=[lemmatizer.lemmatize(y) for y in text]

        return " " .join(text)
    except Exception as e:
        logger.error('some error occured while lemmatization', e)
        raise

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logger.error('some error occured while removing stop words', e)
        raise


def removing_numbers(text):
    try:
        text=''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error('some error occured while removing numbers')
        raise

def lower_case(text):
    try:

        text = text.split()

        text=[y.lower() for y in text]

        return " " .join(text)
    
    except Exception as e:
        logger.error('some error occured while converting to lower case')
        raise

def removing_punctuations(text):
    try:
        ## Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())

        return text.strip()
    except Exception as e:
        logger.error('some error occured while removing punctuations')
        raise

def removing_urls(text):
    try:

        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error('some error occured while removing urls',e)
        raise

def remove_small_sentences(df):
    try:

        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        
    except Exception as e:
        logger.error('failed to remove small sentences')
        raise

def normalize_text(df):
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        return df
    except Exception as e:
        logger.critical('some error occured while applying normatization ',e)
        raise

def export(train_processed_data : pd.DataFrame,test_processed_data : pd.DataFrame) -> None:
    try:
        # store the data inside data/processed
        data_path = os.path.join("data","processed")

        os.makedirs(data_path)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))
        logger.debug('data exported successfully')
    except Exception as e:
        logger.critical('some error occured while exporting data', e)
        raise

def main() -> None:
    try:
        train,test = load_data('./data/raw/train.csv','./data/raw/test.csv')
        train_processed_data = normalize_text(train)
        test_processed_data = normalize_text(test)
        export(train_processed_data,test_processed_data)
        logger.debug('main function executed successfully')
    except Exception as e:
        logger.error('some error occured while executing the main function ', e)
        raise

if __name__ == '__main__':
    main()