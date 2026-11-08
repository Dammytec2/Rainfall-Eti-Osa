
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("ETI-OSA RAINFALL PREDICTION")
print("UPGRADED MACHINE LEARNING MODEL TRAINING")
print("=" * 65)


# --------------------------------------------------
# 1. LOAD PREPARED DATASETS
# --------------------------------------------------

train = pd.read_csv("train_dataset.csv")
validation = pd.read_csv("validation_dataset.csv")
test = pd.read_csv("test_dataset.csv")

target = "TARGET_RAINFALL"

excluded_columns = [
    "DATE",
    "TARGET_DATE",
    "TARGET_RAINFALL",
    "YEAR",
    "DOY",
    "MONTH_NAME",
    "RAINFALL",
"TEMPERATURE",
"HUMIDITY",
"WIND_SPEED",
"PRESSURE"
]

feature_columns = [
    col for col in train.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(train[col])
]

# Check that all datasets contain the same features
for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    missing_features = [
        col for col in feature_columns
        if col not in dataset.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name} dataset is missing features: {missing_features}"
        )

    if dataset[feature_columns].isnull().any().any():
        raise ValueError(f"Missing feature values in {name} dataset.")

    if dataset[target].isnull().any():
        raise ValueError(f"Missing target values in {name} dataset.")


X_train = train[feature_columns]
y_train = train[target]

X_val = validation[feature_columns]
y_val = validation[target]

X_test = test[feature_columns]
y_test = test[target]


print("\nDATASET INFORMATION")
print("=" * 65)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))
print("Number of features:", len(feature_columns))


# --------------------------------------------------
# 2. DEFINE EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(actual, predicted):

    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# 3. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        loss="squared_error",
        random_state=42
    ),

    "Gradient Boosting Huber": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        loss="huber",
        random_state=42
    ),

    "Gradient Boosting Log": TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )
}


# --------------------------------------------------
# 4. TRAIN AND VALIDATE MODELS
# --------------------------------------------------

results = []
trained_models = {}

print("\nTRAINING MODELS")
print("=" * 65)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = evaluate_model(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"]
    })

    trained_models[name] = model

    print("MAE:", round(metrics["MAE"], 4))
    print("RMSE:", round(metrics["RMSE"], 4))
    print("R2 Score:", round(metrics["R2"], 4))


# --------------------------------------------------
# 5. COMPARE AGAINST A SIMPLE BASELINE
# --------------------------------------------------

print("\nBASELINE COMPARISON")
print("=" * 65)

baseline = DummyRegressor(strategy="mean")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

baseline_metrics = evaluate_model(
    y_val,
    baseline_predictions
)

print("Mean Baseline MAE:", round(baseline_metrics["MAE"], 4))
print("Mean Baseline RMSE:", round(baseline_metrics["RMSE"], 4))
print("Mean Baseline R2:", round(baseline_metrics["R2"], 4))


# --------------------------------------------------
# 6. COMPARE VALIDATION RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\nMODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# Select the model with the lowest validation MAE

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nSELECTED MODEL:", best_model_name)


# --------------------------------------------------
# 7. RETRAIN USING TRAINING + VALIDATION DATA
# --------------------------------------------------

X_train_val = pd.concat(
    [X_train, X_val],
    ignore_index=True
)

y_train_val = pd.concat(
    [y_train, y_val],
    ignore_index=True
)

print("\nRETRAINING SELECTED MODEL")
print("=" * 65)

best_model.fit(X_train_val, y_train_val)


# --------------------------------------------------
# 8. FINAL TEST EVALUATION
# --------------------------------------------------

test_predictions = best_model.predict(X_test)

test_predictions = np.maximum(
    test_predictions,
    0
)

test_metrics = evaluate_model(
    y_test,
    test_predictions
)

# Evaluate the baseline on the test dataset too
baseline_test_predictions = baseline.predict(X_test)

baseline_test_metrics = evaluate_model(
    y_test,
    baseline_test_predictions
)

print("\nFINAL TEST RESULTS")
print("=" * 65)

print("Selected model:", best_model_name)

print("Test MAE:",
      round(test_metrics["MAE"], 4), "mm")

print("Test RMSE:",
      round(test_metrics["RMSE"], 4), "mm")

print("Test R2 Score:",
      round(test_metrics["R2"], 4))

print("\nBASELINE TEST RESULTS")

print("Baseline MAE:",
      round(baseline_test_metrics["MAE"], 4), "mm")

print("Baseline RMSE:",
      round(baseline_test_metrics["RMSE"], 4), "mm")

print("Baseline R2:",
      round(baseline_test_metrics["R2"], 4))


# --------------------------------------------------
# 9. EVALUATE DIFFERENT RAINFALL CATEGORIES
# --------------------------------------------------

def rainfall_category(value):

    if value == 0:
        return "No rain"

    elif value <= 2.5:
        return "Light rain"

    elif value <= 10:
        return "Moderate rain"

    elif value <= 50:
        return "Heavy rain"

    else:
        return "Very heavy rain"


category_results = []

actual_categories = y_test.apply(rainfall_category)

for category in [
    "No rain",
    "Light rain",
    "Moderate rain",
    "Heavy rain",
    "Very heavy rain"
]:

    mask = actual_categories == category

    if mask.sum() == 0:
        continue

    actual_values = y_test[mask]

    predicted_values = test_predictions[mask]

    category_mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    category_results.append({
        "Rainfall Category": category,
        "Number of Days": int(mask.sum()),
        "MAE (mm)": category_mae,
        "Average Actual (mm)": actual_values.mean(),
        "Average Predicted (mm)": predicted_values.mean()
    })


category_df = pd.DataFrame(category_results)

print("\nRAINFALL CATEGORY EVALUATION")
print("=" * 65)

print(category_df.to_string(index=False))

category_df.to_csv(
    "rainfall_category_evaluation.csv",
    index=False
)


# --------------------------------------------------
# 10. SAVE DATED PREDICTIONS
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "DATE": pd.to_datetime(test["TARGET_DATE"]),
    "ACTUAL_RAINFALL": y_test.values,
    "PREDICTED_RAINFALL": test_predictions
})

prediction_results["ERROR"] = (
    prediction_results["ACTUAL_RAINFALL"]
    - prediction_results["PREDICTED_RAINFALL"]
)

prediction_results["ABSOLUTE_ERROR"] = (
    prediction_results["ERROR"].abs()
)

prediction_results["ACTUAL_CATEGORY"] = (
    prediction_results["ACTUAL_RAINFALL"]
    .apply(rainfall_category)
)

prediction_results.to_csv(
    "rainfall_predictions.csv",
    index=False
)


# --------------------------------------------------
# 11. SAVE FINAL MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": best_model,
        "features": feature_columns,
        "model_name": best_model_name,
        "test_mae": test_metrics["MAE"],
        "test_rmse": test_metrics["RMSE"],
        "test_r2": test_metrics["R2"]
    },
    "best_rainfall_model.pkl"
)


# --------------------------------------------------
# 12. SAVE EVALUATION REPORT
# --------------------------------------------------

evaluation_report = pd.DataFrame([
    {
        "Model": best_model_name,
        "Dataset": "Test 2026",
        **test_metrics
    },
    {
        "Model": "Mean Baseline",
        "Dataset": "Test 2026",
        **baseline_test_metrics
    }
])

evaluation_report.to_csv(
    "model_evaluation_report.csv",
    index=False
)


# --------------------------------------------------
# 13. COMPLETION SUMMARY
# --------------------------------------------------

print("\nMODEL TRAINING COMPLETED!")
print("=" * 65)

print("Selected model:", best_model_name)

print("\nGenerated files:")
print("- model_comparison.csv")
print("- model_evaluation_report.csv")
print("- rainfall_category_evaluation.csv")
print("- rainfall_predictions.csv")
print("- best_rainfall_model.pkl")
