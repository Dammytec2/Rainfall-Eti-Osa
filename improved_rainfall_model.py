
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 65)
print("IMPROVED RAINFALL PREDICTION - RANDOM FOREST EXPERIMENT")
print("=" * 65)

# 1. Load datasets
train = pd.read_csv("train_dataset.csv")
validation = pd.read_csv("validation_dataset.csv")
test = pd.read_csv("test_dataset.csv")

# 2. Select the same 44 features used by the other models
excluded_columns = [
    "DATE", "YEAR", "DOY",
    "TARGET_DATE", "TARGET_RAINFALL",
    "MONTH_NAME",
    "RAINFALL", "TEMPERATURE",
    "HUMIDITY", "WIND_SPEED", "PRESSURE"
]

feature_columns = [
    col for col in train.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(train[col])
]

X_train = train[feature_columns]
y_train = train["TARGET_RAINFALL"]

X_val = validation[feature_columns]
y_val = validation["TARGET_RAINFALL"]

X_test = test[feature_columns]
y_test = test["TARGET_RAINFALL"]

print(f"\nTraining records: {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Testing records: {len(test)}")
print(f"Input features: {len(feature_columns)}")

# 3. Create rainfall sample weights.
# Higher rainfall receives more weight during training.
def create_weights(y, strength):
    return 1 + strength * np.sqrt(y.clip(lower=0))

# 4. Experiment with different weighting strengths
experiments = {
    "Random Forest Log - Standard": 0.0,
    "Random Forest Log - Balanced": 1.0,
    "Random Forest Log - Heavy Rain Focus": 3.0
}

results = []
trained_models = {}

print("\nTRAINING EXPERIMENTS")
print("-" * 65)

for name, strength in experiments.items():

    print(f"\nTraining {name}...")

    base_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    weights = create_weights(y_train, strength)

    model.fit(
        X_train,
    y_train,
    sample_weight=weights
    )

    predictions = np.maximum(model.predict(X_val), 0)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "Validation_MAE": mae,
        "Validation_RMSE": rmse,
        "Validation_R2": r2
    })

    trained_models[name] = model

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R2: {r2:.4f}")

# 5. Compare models using validation MAE
comparison = pd.DataFrame(results)
comparison = comparison.sort_values("Validation_MAE")

comparison.to_csv(
    "improved_model_comparison.csv",
    index=False
)

print("\nVALIDATION MODEL COMPARISON")
print("-" * 65)
print(comparison.to_string(index=False))

best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

print(f"\nSELECTED MODEL: {best_name}")

# 6. Evaluate the selected model on untouched test data
test_predictions = np.maximum(best_model.predict(X_test), 0)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("\nFINAL TEST RESULTS")
print("-" * 65)
print(f"Test MAE: {test_mae:.4f} mm")
print(f"Test RMSE: {test_rmse:.4f} mm")
print(f"Test R2: {test_r2:.4f}")

# 7. Compare rainfall categories
bins = [-0.001, 0, 2.5, 10, 50, float("inf")]
labels = [
    "No rain",
    "Light rain",
    "Moderate rain",
    "Heavy rain",
    "Very heavy rain"
]

category_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_predictions
})

category_results["Category"] = pd.cut(
    category_results["Actual"],
    bins=bins,
    labels=labels
)

category_results["Absolute_Error"] = abs(
    category_results["Actual"] -
    category_results["Predicted"]
)

print("\nRAINFALL CATEGORY PERFORMANCE")
print("-" * 65)

print(
    category_results.groupby(
        "Category", observed=False
    ).agg(
        Days=("Actual", "count"),
        MAE=("Absolute_Error", "mean"),
        Average_Actual=("Actual", "mean"),
        Average_Predicted=("Predicted", "mean")
    ).round(3).to_string()
)

# 8. Save selected model
joblib.dump(
    {
        "model": best_model,
        "features": feature_columns,
        "model_name": best_name,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    },
    "improved_rainfall_model.pkl"
)

# 9. Save predictions
prediction_output = pd.DataFrame({
    "DATE": test["TARGET_DATE"],
    "ACTUAL_RAINFALL": y_test.values,
    "PREDICTED_RAINFALL": test_predictions
})

prediction_output.to_csv(
    "improved_rainfall_predictions.csv",
    index=False
)

print("\nEXPERIMENT COMPLETED!")
print("Generated files:")
print("- improved_model_comparison.csv")
print("- improved_rainfall_predictions.csv")
print("- improved_rainfall_model.pkl")
