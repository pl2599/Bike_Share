import pandas as pd
import numpy as np
import dill
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Booleans to determine what lines of code to run
train_linear = False
train_rf = False
train_xgb = False
model_compare_results = False
hyper_tune = False
train_champion = True
predict_test = False


# Read Data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print(train.columns)
print(test.columns)

# Define rmsle function
def rmsle(y, pred):
    """FUnction to calculte rmsle

    Args:
        y (pd series): actual y values
        pred (pd series): predicted y values

    Returns:
        float: rmsle score
    """
    return np.sqrt(mean_squared_log_error(y, pred))

# Create scorer to be fed into cv
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

def rmsle_cv_mean(model, folds=5):
    """Obtains the cv rmsle 

    Args:
        model (model or pipeline): Model to be fed into cv
        folds (integer): Number of folds to cv

    Returns:
        float: Average rmsle score 
    """
    scores = cross_val_score(
        model, X_train, y_train,
        cv=folds,
        scoring=rmsle_scorer
    )

    return -1 * scores.mean()

# Remove outliers for train data
def remove_count_outlier(df):
    """Function to remove outliers that are beyond 3 standard deviation for counts

    Args:
        df (pd dataframe): Raw data of bike data

    Returns:
        pd dataframe: dataframe after removing outliers of the count column
    """
    new_df = df[abs(df['count'] - 
                          df['count'].mean()) <= 3 * df['count'].std()]
    return new_df

train = remove_count_outlier(train)

# Split to X_train and y_train
X_train, y_train = train.drop(['count'], axis=1), train['count']

# Create a transformer class to create columns using datetime
class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        """Creates columns based on datetime column

        Args:
            x (pd series): dataframe of x values
            y (pd series, optional): outputs. Defaults to None.

        Returns:
            pd DataFrame: Transformed dataframe with additional columns
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

# Transformer to select columns
all_cols = num_cols + categ_cols + ['datetime']
select_transformer = FunctionTransformer(lambda x: x[all_cols])

# Create a preprocessor transformer for columns, where numerical columns are scaled
preprocessor_scaled = ColumnTransformer(
    transformers = [
        ('datetime', date_transformer, 'datetime'),
        ('num', num_transformer, num_cols),
        ('cat', categ_transformer, categ_cols)
    ]
)

# Create a preprocessor transformer for columns, where numerical columns are not scaled
preprocessor_not_scaled = ColumnTransformer(
    transformers = [
        ('datetime', date_transformer, 'datetime'),
        ('cat', categ_transformer, categ_cols)
    ],
    remainder='passthrough'
)


# Model Comparison

# Linear Regression
if train_linear:

    # Initialize Model
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

    # Save model in output folder
    with open('../output/lm_model', 'wb') as file:
        dill.dump(lm_transformed, file)

else:
    try:
        # Open model in output folder
        with open('../output/lm_model', 'rb') as file:
            lm_transformed = dill.load(file)
    except IOError:
        pass


# Random Forest
if train_rf:

    # Initialize mdoel
    rf = RandomForestRegressor(n_jobs=-1)

    # Create a pipeline to combine the model and preprocessor scaled transformer
    rf_pipeline = Pipeline(
        steps = [
            ('select', select_transformer),
            ('preprocessor', preprocessor_not_scaled),
            ('model', rf)
        ]
    )

    # Train the model
    rf_pipeline.fit(X_train, y_train)

    # Save model in output folder
    with open('../output/rf_model', 'wb') as file:
        dill.dump(rf_pipeline, file)

else:
    try: 
        # Open model in output folder
        with open('../output/rf_model', 'rb') as file:
            rf_pipeline = dill.load(file)
    except IOError:
        pass


# Random Forest
if train_xgb:

    # Initialize mdoel
    xgb = XGBRegressor(n_jobs=-1)

    # Create a pipeline to combine the model and preprocessor scaled transformer
    xgb_pipeline = Pipeline(
        steps = [
            ('select', select_transformer),
            ('preprocessor', preprocessor_not_scaled),
            ('model', xgb)
        ]
    )

    # Transform the y column using TransformedTargetRegressor (required since w/o step will provide negative values)
    xgb_transformed = TransformedTargetRegressor(
        regressor=xgb_pipeline,
        func=np.log1p, 
        inverse_func=np.expm1
    )

    # Train the model
    xgb_transformed.fit(X_train, y_train)

    # Save model in output folder
    with open('../output/xgb_model', 'wb') as file:
        dill.dump(xgb_transformed, file)

else:
    try:
        # Open model in output folder
        with open('../output/xgb_model', 'rb') as file:
            xgb_transformed = dill.load(file)
    except IOError:
        pass

# Model results Comparison
if model_compare_results:

    # Predict and get rmsle for the three models
    lm_rmsle = rmsle_cv_mean(lm_transformed)
    rf_rmsle = rmsle_cv_mean(rf_pipeline)
    xgb_rmsle = rmsle_cv_mean(xgb_transformed)


    # Model Comparison results
    model_results = pd.DataFrame({
        'Models': ['Linear', 'Random_Forest', 'XGBoost'],
        'RMLSE': [lm_rmsle, rf_rmsle, xgb_rmsle]
    })

    print(model_results)


# Hyperparameter Tuning for XGBoost
if hyper_tune:

    # Initialize model
    xgb_2 = XGBRegressor()

    # Create a pipeline to combine the model and preprocessor scaled transformer
    xgb_pipeline_2 = Pipeline(
        steps = [
            ('select', select_transformer),
            ('preprocessor', preprocessor_not_scaled),
            ('model', xgb_2)
        ]
    )

    # Transform the y column using TransformedTargetRegressor (required since w/o step will provide negative values)
    xgb_transformed_2 = TransformedTargetRegressor(
        regressor=xgb_pipeline_2,
        func=np.log1p, 
        inverse_func=np.expm1
    )

    # Define parameter grid 
    param_grid = [{
        'regressor__model__max_depth': range(4, 13, 2),
        'regressor__model__min_child_weight': range(1, 8, 2),
        'regressor__model__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'regressor__model__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8],
        'regressor__model__learning_rate': [0.05, 0.1, 0.2]
    }]

    # Initialize Grid_search
    xgb_grid = GridSearchCV(
        estimator = xgb_transformed_2,
        param_grid=param_grid,
        cv = 5,
        scoring=rmsle_scorer,
        n_jobs = -1,
        verbose = 1
    )

    # Train the Model
    xgb_grid.fit(X_train, y_train)

    # Print Score and Best Parameters
    print(f"The best RMSLE score is: { -1 * xgb_grid.best_score_:.5f} for the following parameters:")
    print(xgb_grid.best_params_)

    # Save model in output folder
    with open('../output/xgb_grid', 'wb') as file:
        dill.dump(xgb_grid, file)

else:
    try:
        # Open model in output folder
        with open('../output/xgb_grid', 'rb') as file:
            xgb_grid = dill.load(file)
    except IOError:
        pass


# Train Champion
if train_champion:
    # Initialize model
    champion = XGBRegressor(n_jobs=-1, **xgb_grid.best_params_)

    # Create a pipeline to combine the model and preprocessor scaled transformer
    champion_pipeline = Pipeline(
        steps = [
            ('select', select_transformer),
            ('preprocessor', preprocessor_not_scaled),
            ('model', champion)
        ]
    )

    # Transform the y column using TransformedTargetRegressor (required since w/o step will provide negative values)
    champion_transformed = TransformedTargetRegressor(
        regressor=champion_pipeline,
        func=np.log1p, 
        inverse_func=np.expm1
    )

    # Train the Model
    champion_transformed.fit(X_train, y_train)

   # Save model in output folder
    with open('../output/champion', 'wb') as file:
        dill.dump(champion_transformed, file)

else:
    try:
        # Open model in output folder
        with open('../output/champion', 'rb') as file:
            champion_transformed = dill.load(file)
    except IOError:
        pass

# Make predictions on test data
if predict_test:
    # Make Predictions
    predictions = champion_transformed.predict(test)

    # Export Submission
    submission = pd.DataFrame({
        "datetime": test['datetime'],
        "count": predictions
    })

    submission.to_csv('../output/bike_predictions.csv', index=False)
