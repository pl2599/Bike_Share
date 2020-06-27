import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

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

# Create a transformer class to create columns using datetime
class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, x, y=None):
        """Creates columns based on datetime column

        Args:
            x ([pd series]): dataframe of x values
            y ([pd series], optional): outputs. Defaults to None.

        Returns:
            [pd DataFrame]: Transformed dataframe with additional columns
        """
        x_datetime = x.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        return pd.DataFrame({
            'hour': x_datetime.apply(lambda x: x.hour),
            'month': x_datetime.apply(lambda x: x.month),
            'year': x_datetime.apply(lambda x: x.year),
        })



# Transformer for datetime
date_transformer = Pipeline(
    steps = [
        ('datetime', DateTransformer()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

print(date_transformer.fit_transform(train['datetime']))