import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from config.config import (FILEPATH_DATA_RAW,
                           FILEPATH_DATA_PROCESSED,
                           FILEPATH_ARGS)
from src.utils.utils import load_json


def num_bathroom_from_text(text: str) -> str:
    """Get number of bathrooms from `bathrooms_text`"""
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        else:
            return np.NaN
    except ValueError:
        return np.NaN


def select_subset(df: DataFrame) -> DataFrame:
    """Select small subset of columns."""
    cols_subset = load_json(str(FILEPATH_ARGS))['cols_subset']
    df_subset = df[cols_subset].copy()
    df_subset.rename(columns={'neighbourhood_group_cleansed': 'neighbourhood'}, inplace=True)
    return df_subset


def price_drop_listings(df: DataFrame) -> DataFrame:
    """Remove the listings where price feature is between 0 and 10 dollars."""
    df['price'] = df['price'].str.extract(r"(\d+).")
    try:
        df['price'] = df['price'].astype(int)
        return df[df['price'] >= 10]
    except ValueError as e:
        print('Some string value/s not have digit characters to extract: ', e)


def category_feature(price: Series) -> Series:
    """Create `category` feature from `price` feature. """
    return pd.cut(price, bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])


def preprocess_amenities_column(df: DataFrame) -> DataFrame:
    """Extract column information for amenities."""
    df['TV'] = df['amenities'].str.contains('TV')
    df['TV'] = df['TV'].astype(int)
    df['Internet'] = df['amenities'].str.contains('Internet')
    df['Internet'] = df['Internet'].astype(int)
    df['Air_conditioning'] = df['amenities'].str.contains('Air conditioning')
    df['Air_conditioning'] = df['Air_conditioning'].astype(int)
    df['Kitchen'] = df['amenities'].str.contains('Kitchen')
    df['Kitchen'] = df['Kitchen'].astype(int)
    df['Heating'] = df['amenities'].str.contains('Heating')
    df['Heating'] = df['Heating'].astype(int)
    df['Wifi'] = df['amenities'].str.contains('Wifi')
    df['Wifi'] = df['Wifi'].astype(int)
    df['Elevator'] = df['amenities'].str.contains('Elevator')
    df['Elevator'] = df['Elevator'].astype(int)
    df['Breakfast'] = df['amenities'].str.contains('Breakfast')
    df['Breakfast'] = df['Breakfast'].astype(int)

    df.drop('amenities', axis=1, inplace=True)
    
    return df


def feature_engineering(df: DataFrame, data_subset: bool =True) -> DataFrame:
    """Create or modify features from the existing ones."""
    df_feat_eng = df.copy()
    df_feat_eng['bathrooms'] = df_feat_eng['bathrooms_text'].map(num_bathroom_from_text)
    if data_subset:
        df_feat_eng = select_subset(df_feat_eng)
    df_feat_eng.dropna(axis=0, inplace=True)
    df_feat_eng = price_drop_listings(df_feat_eng)
    df_feat_eng['category'] = category_feature(df_feat_eng['price'])
    df_feat_eng = preprocess_amenities_column(df_feat_eng)

    return df_feat_eng


def preprocess() -> DataFrame:
    """Preprocess raw DataFrame."""
    df_processed = pd.read_csv(str(FILEPATH_DATA_RAW), low_memory=False)
    df_processed = feature_engineering(df_processed, data_subset=True)
    df_processed.to_csv(str(FILEPATH_DATA_PROCESSED))

    return df_processed


if __name__ == '__main__':
    preprocess()
