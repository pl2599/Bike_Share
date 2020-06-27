import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer

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
X, y = train.drop(['count'], axis=1), train['count']

# Define rmlse scorer
def rmlse(y, pred):
    return np.sqrt(mean_squared_log_error(y, pred))

# Create scorer to be fed into cv
rmlse_scorer = make_scorer(rmlse, greater_is_better=False)


# Create a transformer class to create columns using datetime
class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
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

# Transformer for numerical columns
num_cols = ['atemp', 'humidity']
num_transformer = StandardScaler()

# Transformer for categoriacl columns
categ_cols = ['weather', 'season', 'workingday']
categ_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor transformer for columns
preprocessor = ColumnTransformer(
    transformers = [
        ('datetime', date_transformer, 'datetime'),
        ('num', num_transformer, num_cols),
        ('cat', categ_transformer, categ_cols)
    ]
)

# Linear Regression
lm = LinearRegression()

# Create a pipeline to combine the model and preprocessor transformer
lm_pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('model', lm)
    ]
)

# Transform the y column
lm_transformed = TransformedTargetRegressor(
    regressor=lm_pipeline,
    func=np.log1p, 
    inverse_func=np.expm1
)

# Train the model
lm_transformed.fit(X, y)

# Predict
lm_scores = cross_val_score(
    lm_transformed, X, y,
    cv = 5,
    scoring=rmlse_scorer
)

lm_rmlse = -1 * lm_scores.mean()
print(f"The average rmlse after cross validation is: {lm_rmlse}")

