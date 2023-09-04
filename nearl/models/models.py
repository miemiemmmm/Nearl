import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

__all__ = [
  "rfscore",
]

# Create a random forest regressor.
class rfscore(RandomForestRegressor):
  def __init__(self, n_jobs=-1):
    mtry = 21
    super().__init__(
      n_estimators = 500,
      oob_score=True,
      n_jobs=n_jobs,
      max_features=mtry,
      bootstrap=True,
      min_samples_split=2,
      # min_samples_leaf=1,
      max_depth=20,
      # *args,
      # **kwargs
    )
  def save(self, filename):
    joblib.dump(self, filename)
  def load(self, filename):
    self = joblib.load(filename)


