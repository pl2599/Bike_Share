import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer

# Read Data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# Define rmlse scorer
def rmlse(y, pred):
    return np.sqrt(mean_squared_log_error(y, pred))

# Remove outliers for train data
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

# Split to X_train and y_train
X_train, y_train = train.drop(['count'], axis=1), train['count']

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

# Transformer for datetime - Splitting and OH Encoding
date_transformer = Pipeline(
    steps = [
        ('datetime', DateTransformer()),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Transformer for numerical columns - Scaling
num_cols = ['atemp', 'humidity']
num_transformer = StandardScaler()

# Transformer for categorial columns - OneHot Encoding
categ_cols = ['weather', 'season', 'workingday']
categ_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor transformer for columns, where numerical columns are scaled
preprocessor_scaled = ColumnTransformer(
    transformers = [
        ('datetime', date_transformer, 'datetime'),
        ('num', num_transformer, num_cols),
        ('cat', categ_transformer, categ_cols)
    ]
)

# Linear Regression
lm = LinearRegression()

# Create a pipeline to combine the model and preprocessor scaled transformer
lm_pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor_scaled),
        ('model', lm)
    ]
)

# Transform the y column using TransformedTargetRegressor
lm_transformed = TransformedTargetRegressor(
    regressor=lm_pipeline,
    func=np.log1p, 
    inverse_func=np.expm1
)

# Train the model
lm_transformed.fit(X_train, y_train)

# Predict
lm_scores = cross_val_score(
    lm_transformed, X_train, y_train,
    cv = 5,
    scoring=rmlse_scorer
)

lm_rmlse = -1 * lm_scores.mean()
print(f"The average rmlse after cross validation is: {lm_rmlse}")


# Random Forest

# Transformer to select columns
all_cols = num_cols + categ_cols + ['datetime']
select_transformer = FunctionTransformer(lambda x: x[all_cols])

# Create a preprocessor transformer for columns, where numerical columns are not scaled
preprocessor_not_scaled = ColumnTransformer(
    transformers = [
        ('datetime', date_transformer, 'datetime'),
        ('cat', categ_transformer, categ_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline to combine the model and preprocessor scaled transformer
custom_pipeline = Pipeline(
    steps = [
        ('select', select_transformer),
        ('preprocessor', preprocessor_not_scaled),
    ]
)
