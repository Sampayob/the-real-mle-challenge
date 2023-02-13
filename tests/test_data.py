import os

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data.processed import preprocess
from config.config import (DIR_DATA_PROCESSED,
                            FILEPATH_DATA_PROCESSED)


def test_preprocess():
       "Test `preprocessed_listings.csv' file generation."
       preprocess()
       assert 'preprocessed_listings.csv' in os.listdir(DIR_DATA_PROCESSED), \
              '`preprocessed_listings.csv` was not generated.'

       data_processed = pd.read_csv(str(FILEPATH_DATA_PROCESSED), index_col=0)
       assert not data_processed.empty, '`preprocessed_listings.csv` was generated but its empty.'

       assert 'bathrooms' in data_processed.columns, '`bathrooms` col not in DataFrame.'
       assert 'category' in data_processed.columns, '`category` col not in DataFrame.'
       assert is_numeric_dtype(data_processed['price']), '`price` col was not converted to a numeric type.' 
       assert all(col in data_processed.columns for col in ['TV',
                                                        'Internet',
                                                        'Air_conditioning',
                                                        'Kitchen',
                                                        'Heating',
                                                        'Wifi',
                                                        'Elevator',
                                                        'Breakfast']), \
                                                        'All or some cols created from `amenities` col are missing.'
