import pandas as pd
import numpy as np
from datetime import datetime

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

def remove_count_outlier(df):
    """Function to remove outliers that are beyond 3 standard deviation for counts

    Args:
        df ([pd dataframe]): Raw data of bike data

    Returns:
        [pd dataframe]: dataframe after removing outliers of the count column
    """
    new_df = df[abs(df['count'] - 
                          df['count'].mean()) <= 3 * df['count'].std()]
    return new_df

train = remove_count_outlier(train)

def add_time_date(df):
    """Function to add new columns using the datetime column

    Args:
        df ([pd dataframe]): Raw data of bike data

    Returns:
        [pd dataframe]: Returns dataframe after appending datetime columns
    """
    new_df = df
    new_df.datetime = df['datetime'].apply(lambda x: 
                                                  datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    new_df['hour'] = df['datetime'].apply(lambda x: x.hour)
    new_df['day'] = df['datetime'].apply(lambda x: x.day)
    new_df['month'] = df['datetime'].apply(lambda x: x.month)
    new_df['year'] = df['datetime'].apply(lambda x: x.year)
    
    return new_df


def preprocess_data(df):
    """Function to pre-process the bike share data.
    Adds datetime columns
    Normalize Data


    Args:
        df ([pd dataframe]): Raw data of bike data

    Returns:
        [pd dataframe]: Dataframe after performing preprocessing
    """
    new_df = add_time_date(df)

    return new_df

train = preprocess_data(train)