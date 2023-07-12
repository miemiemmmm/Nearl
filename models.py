from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import joblib

# Create a random forest regressor.
class rfscore(RandomForestRegressor):
  def __init__(*args, **kwargs):
    super().__init__(*args, **kwargs)
  def save(self, filename):
    joblib.dump(self, filename)
  def load(self, filename):
    self = joblib.load(filename)





