
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=" * 60)
print("TWO-STAGE RAINFALL PREDICTION MODEL")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

train = pd.read_csv("train_dataset.csv")
validation = pd.read_csv("validation_dataset.csv")
test = pd.read_csv("test_dataset.csv")

feature_columns = [
    col for col in train.columns
    if col not in [
        "DATE", "YEAR", "DOY",
        "TARGET_DATE", "TARGET_RAINFALL",
        "MONTH_NAME",
        "RAINFALL", "TEMPERATURE",
        "HUMIDITY", "WIND_SPEED",
        "PRESSURE"
    ]
    and pd.api.types.is_numeric_dtype(train[col])
]

X_train = train[feature_columns]
y_train = train["TARGET_RAINFALL"]

X_val = validation[feature_columns]
y_val = validation["TARGET_RAINFALL"]

X_test = test[feature_columns]
y_test = test["TARGET_RAINFALL"]

print("\nDATASET INFORMATION")
print("-" * 40)
print("Training records:", len(train))
print("Validation records:", len(validation))
print("Testing records:", len(test))
print("Input features:", len(feature_columns))

# --------------------------------------------------
# 2. STAGE ONE: RAIN OCCURRENCE CLASSIFIER
# --------------------------------------------------

print("\nSTAGE ONE: TRAINING RAIN CLASSIFIER")
print("-" * 40)

# Rainfall above zero means rain occurred.
y_train_rain = (y_train > 0).astype(int)
y_val_rain = (y_val > 0).astype(int)
y_test_rain = (y_test > 0).astype(int)

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

classifier.fit(X_train, y_train_rain)

# Use validation probabilities to select a threshold.
val_probabilities = classifier.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in np.arange(0.20, 0.81, 0.05):
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    recall = recall_score(
        y_val_rain,
        val_predictions,
        zero_division=0
    )

    precision = precision_score(
        y_val_rain,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val_rain,
        val_predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

threshold_table = pd.DataFrame(threshold_results)

# Choose the threshold with the best validation F1 score.
best_threshold = threshold_table.loc[
    threshold_table["f1"].idxmax(),
    "threshold"
]

print("Selected rain threshold:", round(best_threshold, 2))

print("\nRAIN CLASSIFICATION VALIDATION RESULTS")
print("Accuracy:", round(accuracy_score(
    y_val_rain,
    (val_probabilities >= best_threshold).astype(int)
), 4))

print("Precision:", round(threshold_table["precision"].max(), 4))
print("Recall:", round(
    threshold_table.loc[
        threshold_table["threshold"] == best_threshold,
        "recall"
    ].iloc[0], 4
))
print("F1 Score:", round(
    threshold_table.loc[
        threshold_table["threshold"] == best_threshold,
        "f1"
    ].iloc[0], 4
))

# --------------------------------------------------
# 3. STAGE TWO: RAINFALL AMOUNT REGRESSOR
# --------------------------------------------------

print("\nSTAGE TWO: TRAINING RAINFALL AMOUNT MODEL")
print("-" * 40)

# Train only on days when rainfall actually occurred.
rainy_train_mask = y_train > 0

X_train_wet = X_train.loc[rainy_train_mask]
y_train_wet = y_train.loc[rainy_train_mask]

amount_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    loss="huber",
    random_state=42
)

amount_model.fit(X_train_wet, y_train_wet)

print("Rainy training records:", len(X_train_wet))
print("Rainfall amount model trained successfully.")

# --------------------------------------------------
# 4. TWO-STAGE PREDICTION FUNCTION
# --------------------------------------------------

def predict_two_stage(X):
    probabilities = classifier.predict_proba(X)[:, 1]

    rain_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    amounts = amount_model.predict(X)

    amounts = np.maximum(amounts, 0)

    final_predictions = np.where(
        rain_predictions == 1,
        amounts,
        0
    )

    return final_predictions, probabilities


# --------------------------------------------------
# 5. COMPARE AGAINST EXISTING MODEL
# --------------------------------------------------

print("\nVALIDATION MODEL COMPARISON")
print("-" * 40)

two_stage_val, val_rain_probabilities = predict_two_stage(X_val)

saved_model = joblib.load("best_rainfall_model.pkl")
existing_model = saved_model["model"]

existing_val = np.maximum(
    existing_model.predict(X_val),
    0
)

two_stage_mae = mean_absolute_error(
    y_val,
    two_stage_val
)

existing_mae = mean_absolute_error(
    y_val,
    existing_val
)

two_stage_rmse = np.sqrt(
    mean_squared_error(y_val, two_stage_val)
)

existing_rmse = np.sqrt(
    mean_squared_error(y_val, existing_val)
)

print("Existing model MAE:", round(existing_mae, 4))
print("Two-stage model MAE:", round(two_stage_mae, 4))

print("\nExisting model RMSE:", round(existing_rmse, 4))
print("Two-stage model RMSE:", round(two_stage_rmse, 4))

print("\nTwo-stage validation R2:", round(
    r2_score(y_val, two_stage_val), 4
))

# --------------------------------------------------
# 6. FINAL TEST EVALUATION
# --------------------------------------------------

print("\nFINAL TEST EVALUATION")
print("-" * 40)

two_stage_test, test_rain_probabilities = predict_two_stage(X_test)

existing_test = np.maximum(
    existing_model.predict(X_test),
    0
)

print("Existing model Test MAE:",
      round(mean_absolute_error(y_test, existing_test), 4))

print("Two-stage model Test MAE:",
      round(mean_absolute_error(y_test, two_stage_test), 4))

print("Existing model Test RMSE:",
      round(np.sqrt(mean_squared_error(y_test, existing_test)), 4))

print("Two-stage model Test RMSE:",
      round(np.sqrt(mean_squared_error(y_test, two_stage_test)), 4))

print("Two-stage Test R2:",
      round(r2_score(y_test, two_stage_test), 4))

# Classification test metrics.
test_class_predictions = (
    test_rain_probabilities >= best_threshold
).astype(int)

print("\nRAIN OCCURRENCE TEST RESULTS")
print("Accuracy:", round(accuracy_score(
    y_test_rain, test_class_predictions
), 4))

print("Precision:", round(precision_score(
    y_test_rain, test_class_predictions, zero_division=0
), 4))

print("Recall:", round(recall_score(
    y_test_rain, test_class_predictions, zero_division=0
), 4))

print("F1 Score:", round(f1_score(
    y_test_rain, test_class_predictions, zero_division=0
), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_rain, test_class_predictions))

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

comparison = pd.DataFrame({
    "DATE": pd.to_datetime(test["TARGET_DATE"]),
    "ACTUAL_RAINFALL": y_test.values,
    "EXISTING_MODEL_PREDICTION": existing_test,
    "TWO_STAGE_PREDICTION": two_stage_test,
    "RAIN_PROBABILITY": test_rain_probabilities
})

comparison.to_csv(
    "two_stage_predictions.csv",
    index=False
)

evaluation = pd.DataFrame([
    {
        "Model": "Existing Gradient Boosting Log",
        "Validation_MAE": existing_mae,
        "Validation_RMSE": existing_rmse
    },
    {
        "Model": "Two-Stage Model",
        "Validation_MAE": two_stage_mae,
        "Validation_RMSE": two_stage_rmse
    }
])

evaluation.to_csv(
    "two_stage_model_comparison.csv",
    index=False
)

joblib.dump(
    {
        "classifier": classifier,
        "amount_model": amount_model,
        "threshold": best_threshold,
        "features": feature_columns
    },
    "two_stage_rainfall_model.pkl"
)

print("\nGENERATED FILES")
print("- two_stage_predictions.csv")
print("- two_stage_model_comparison.csv")
print("- two_stage_rainfall_model.pkl")

print("\nTWO-STAGE MODEL EXPERIMENT COMPLETED!")
print("=" * 60)
