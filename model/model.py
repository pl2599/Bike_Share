import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixMin

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

    return pd.get_dummies(df, prefix=[var[0] for var in categ_variables], drop_first=True, columns=categ_variables)


# Create a transformer class to create columns using datetime
class DateTransformer(BaseEstimator, TransformerMixMin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, x, y=None):
        """Creates columns based on datetime column

        Args:
            x ([pd series]): dataframe of x values
            y ([pd series], optional): outputs. Defaults to None.

        Returns:
            [pd dataframe]: Transformed dataframe with additional columns
        """
        x_datetime = x.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        return pd.dataframe({
            'hour': x_datetime.apply(lambda x: x.hour),
            'month': x_datetime.apply(lambda x: x.month),
            'year': x_datetime.apply(lambda x: x.year),
        })



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

    # Add datetime split columns
    new_df = add_time_date(df)

    # OH Encode Categ Variables
    new_df = get_categ_dummies(new_df, categ_variables)

    # Drop unnecessary Columns
    new_df = new_df.drop(drop_variables, axis=1)

    return new_df.drop('count', axis = 1), new_df['count']


X_train, y_train = preprocess_data(train)




