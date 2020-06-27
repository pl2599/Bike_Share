import pandas as pd
import numpy as np
from datetime import datetime
#from sklearn.base import BaseEstimator, TransformerMixMin

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print(train[['count', 'season']])