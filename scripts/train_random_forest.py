from sklearn.ensemble import RandomForestRegressor as randomforest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np

from BetaPose import utils, data_io, models


print("Loading data...")
with data_io.hdf_operator("/media/yzhang/MieT5/BetaPose/tests/test_randomforest.h5") as h5file:
  rf_training_data = h5file.data("rf")
  label_training_data = h5file.data("label").ravel()

print("Data loaded.")



# Split your data into training and testing sets.
rf_model = models.rfscore(n_estimators=100, random_state=42);
X_train, X_test, y_train, y_test = train_test_split(rf_training_data, label_training_data, test_size=0.2, random_state=42)



# Fit the regressor with the training data.
rf_model.fit(X_train, y_train)

# Predict on the test data.
y_pred = rf_model.predict(X_test)

# Compute the mean squared error of your predictions.
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
