import time

import numpy as np
from Nearl import data_io, log, models
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st = time.perf_counter()
log("Loading data...")

input_files = [
    "/MieT5/Nearl/data/trainingdata/misato_trainset_randomforest.h5",
    # "/MieT5/Nearl/data/trainingdata/misato_randomforest.h5",
    # "/MieT5/Nearl/data/trainingdata/pdbbindrefined_v2016_randomforest.h5",
    # "/MieT5/Nearl/data/trainingdata/misato_randomforest_step10.h5",
    # "/MieT5/Nearl/data/trainingdata/misato_testset_randomforest.h5",
]

rf_data = []
label_data = []
for input_hdfile in input_files:
    with data_io.hdf_operator(input_hdfile, "r") as h5file:
        rf_data.append(h5file.data("rf"))
        label_data.append(h5file.data("label").ravel())
rf_training_data = np.concatenate(rf_data, axis=0)
label_training_data = np.concatenate(label_data, axis=0)
print(
    f"Training dataset: {rf_training_data.shape} ; Label number: {len(label_training_data)}"
)
# Load test dataset;
testset_file = "/MieT5/Nearl/data/trainingdata/misato_testset_randomforest.h5"
with data_io.hdf_operator(testset_file, "r") as h5file:
    h5file.draw_structure()
    rf_testset = h5file.data("rf")
    label_testset = h5file.data("label").ravel()

log("Data loaded!!! Good luck!!!")

# Split your data into training and testing sets.
# E.G. 75% training, 25% testing.
# Use train_test_split for separating the training/test data
ratio_test = 0.5
X_train, X_test, y_train, y_test = train_test_split(
    rf_training_data, label_training_data, test_size=ratio_test, random_state=42
)

# X_train = rf_training_data
# X_test  = rf_testset
# y_train = label_training_data
# y_test  = label_testset


rf_regressor = models.rfscore()
# Fit the regressor with the training data.
rf_regressor.fit(X_train, y_train)

log(f"Training time: {time.perf_counter() - st:.4f} seconds")

log("Summary of the model: ")
# print(f"estimators: {rf_regressor.estimators_}")
log(f"estimator: {rf_regressor.estimator_}")
log(f"feature_importances: {rf_regressor.feature_importances_}")
log(f"n_features_in_: {rf_regressor.n_features_in_}")
# print(f"base_estimator: {rf_regressor.base_estimator_}")
# print(f"feature_names_in_: {rf_regressor.feature_names_in_}")
log(f"n_outputs: {rf_regressor.n_outputs_}")
log(f"oob_score: {rf_regressor.oob_score_}")
log(f"oob_prediction: {rf_regressor.oob_prediction_}")

print("###############################################################")
# Result of the prediction on the training data.
y_pred = rf_regressor.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred)
r2_train = r2_score(y_train, y_pred)
print("On Training set: ")
print(
    f"Mean squared error: {mse_train:.3f} ; RMSE: {np.sqrt(mse_train):.3f}, R^2: {r2_train:.3f}"
)
print(
    f"Median of residuals: {np.median(y_train - y_pred):.3f}, Median of absolute residuals: {np.median(np.abs(y_train - y_pred)):.3f}"
)
pearson_corr, p_value = stats.pearsonr(y_train, y_pred)
spearman_corr, _ = stats.spearmanr(y_train, y_pred)
kendall_tau, _ = stats.kendalltau(y_train, y_pred)
print(
    f"Pearson {pearson_corr:.3f}, Spearman {spearman_corr:.3f}, Kendall {kendall_tau:.3f}"
)


# y_true represents the true values
# y_pred represents the predicted values from your model

print("###############################################################")
# Predict on the test data.
y_pred = rf_regressor.predict(X_test)
# Compute the mean squared error of your predictions.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("On test dataset: ")
print(f"Mean squared error: {mse:.3f} ; RMSE: {np.sqrt(mse):.3f}, R^2: {r2:.3f}")
print(
    f"Median of residuals: {np.median(y_test - y_pred):.3f}, Median of absolute residuals: {np.median(np.abs(y_test - y_pred)):.3f}"
)
pearson_corr, p_value = stats.pearsonr(y_test, y_pred)
spearman_corr, _ = stats.spearmanr(y_test, y_pred)
kendall_tau, _ = stats.kendalltau(y_test, y_pred)
print(f"Pearson {pearson_corr}, Spearman {spearman_corr}, Kendall {kendall_tau}")


print("###############################################################")

oob_predictions = rf_regressor.oob_prediction_
mse_oob = mean_squared_error(y_train, oob_predictions)
r2_oob = r2_score(y_train, oob_predictions)
print("On OOB set: ")
print(
    f"Mean squared error: {mse_oob:.3f} ; RMSE: {np.sqrt(mse_oob):.3f}, R^2: {r2_oob:.3f}"
)
print(
    f"Median of residuals: {np.median(y_train - oob_predictions):.3f}, Median of absolute residuals: {np.median(np.abs(y_train - oob_predictions)):.3f}"
)
pearson_corr, _ = stats.pearsonr(y_train, oob_predictions)
spearman_corr, _ = stats.spearmanr(y_train, oob_predictions)
kendall_tau, _ = stats.kendalltau(y_train, oob_predictions)
print(f"Pearson {pearson_corr}, Spearman {spearman_corr}, Kendall {kendall_tau}")
# print(f"Pearson correlation coefficient: {pearson_correlation_coefficient} ; p_value {p_value}");

print("###############################################################")
