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

# Remove outliers for train data
train = remove_count_outlier(train)

def add_time_date(df):
    """Function to add new columns using the datetime column

    Args:
        df ([pd dataframe]): Dataframe with datetime column

    Returns:
        [pd dataframe]: Returns dataframe after appending datetime columns
    """
    new_df = df
    new_df['datetime'] = df['datetime'].apply(lambda x: 
                                                  datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    new_df['hour'] = df['datetime'].apply(lambda x: x.hour)
    new_df['month'] = df['datetime'].apply(lambda x: x.month)
    new_df['year'] = df['datetime'].apply(lambda x: x.year)
    
    return new_df

def get_categ_dummies(df, categ_variables):
    """Function to get all category dummy variables, dropping the first column

    Args:
        df ([pd dataframe]): Dataframe with category variables
        categ_variables ([type]): list of category variables
    """

    dummy_df = pd.get_dummies(df, prefix=[var[0] for var in categ_variables], drop_first=True, columns=categ_variables)
    print(dummy_df.columns)
    # dummy_df = dummy_df.drop(categ_variables, axis=1)
    return pd.concat([df, dummy_df], axis=1)



def preprocess_data(df):
    """Function to pre-process the bike share data.
    Adds datetime columns
    Adds Dummy variables
    Drops Unnecessary Columns


    Args:
        df ([pd dataframe]): Raw data of bike data

    Returns:
        [pd dataframe]: Dataframe after performing preprocessing
    """

    categ_variables = ['weather', 'season', 'workingday', 'hour', 'month', 'year']
    drop_variables = ['datetime', 'registered', 'casual', 'windspeed', 'temp']

    new_df = add_time_date(df)
    new_df = get_categ_dummies(new_df, categ_variables)

    new_df = new_df.drop(drop_variables, axis=1)

    return new_df

train = preprocess_data(train)

print(train.columns)