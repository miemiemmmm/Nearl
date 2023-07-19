from sklearn.ensemble import RandomForestRegressor as randomforest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import numpy as np

from BetaPose import utils, data_io, models, printit
import time

st = time.perf_counter();
printit("Loading data...")

input_files = ["/media/yzhang/MieT5/BetaPose/data/trainingdata/misato_randomforest.h5"]

rf_data = [];
label_data = [];
for input_hdfile in input_files:
  with data_io.hdf_operator(input_hdfile, read_only=True) as h5file:
    rf_data.append(h5file.data("rf"))
    label_data.append(h5file.data("label").ravel())
rf_training_data = np.concatenate(rf_data, axis=0)
label_training_data = np.concatenate(label_data, axis=0)
printit("Data loaded!!! Good luck!!!");

# Split your data into training and testing sets.
# E.G. 75% training, 25% testing.
ratio_test = 0.25
X_train, X_test, y_train, y_test = train_test_split(rf_training_data, label_training_data, test_size=ratio_test, random_state=42)


rf_regressor = models.rfscore();
# Fit the regressor with the training data.
rf_regressor.fit(X_train, y_train)

printit(f"Training time: {time.perf_counter()-st:.4f} seconds")

printit(f"Summary of the model: ")
# print(f"estimators: {rf_regressor.estimators_}")
printit(f"estimator: {rf_regressor.estimator_}")
printit(f"feature_importances: {rf_regressor.feature_importances_}")
printit(f"n_features_in_: {rf_regressor.n_features_in_}")
# print(f"base_estimator: {rf_regressor.base_estimator_}")
# print(f"feature_names_in_: {rf_regressor.feature_names_in_}")
printit(f"n_outputs: {rf_regressor.n_outputs_}")
printit(f"oob_score: {rf_regressor.oob_score_}")
printit(f"oob_prediction: {rf_regressor.oob_prediction_}")


# Predict on the training data.
y_pred = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred)
r2_train = r2_score(y_train, y_pred)

from scipy.stats import pearsonr

correlation_matrix = np.corrcoef(y_train, y_pred)
print("Debug: correlation_matrix: ", correlation_matrix)
pearson_correlation_coefficient = correlation_matrix[0, 1]
print("Debug: pearson_correlation_coefficient: ", pearson_correlation_coefficient)
_pearson_correlation_coefficient, p_value = pearsonr(y_train, y_pred)
print("Debug: pearson_correlation_coefficient: ", _pearson_correlation_coefficient, "p_value: ", p_value)
pearson_coeff = np.sqrt(r2_train)
print("Debug: pearson_coeff: ", pearson_coeff)
print(np.isclose(pearson_correlation_coefficient, pearson_coeff), np.isclose(_pearson_correlation_coefficient, pearson_coeff))



print(f"On Training set: Mean squared error: {mse_train:.3f} ; RMSE: {np.sqrt(mse_train):.3f}, R^2: {r2_train:.3f}")


# y_true represents the true values
# y_pred represents the predicted values from your model


# Predict on the test data.
y_pred = rf_regressor.predict(X_test)
# Compute the mean squared error of your predictions.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"On Test set: Mean squared error: {mse:.3f} ; RMSE: {np.sqrt(mse):.3f}, R^2: {r2:.3f}")

