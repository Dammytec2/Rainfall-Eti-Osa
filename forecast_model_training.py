import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 65)
print("ETI-OSA FORECAST-BASED RAINFALL MODEL")
print("=" * 65)

# 1. Load historical NASA weather data
data = pd.read_csv("eti_osa_weather_master.csv")

data["DATE"] = pd.to_datetime(data["DATE"])
data = data.sort_values("DATE").reset_index(drop=True)

# 2. Convert units to match Open-Meteo
# NASA wind speed: m/s
# Open-Meteo wind speed: km/h
# NASA pressure: kPa
# Open-Meteo pressure: hPa

data["WIND_SPEED"] = data["WIND_SPEED"]  # Keep in m/s
data["PRESSURE"] = data["PRESSURE"]      # Keep in kPa

# 3. Create calendar features
data["MONTH"] = data["DATE"].dt.month
data["DAY_OF_YEAR"] = data["DATE"].dt.dayofyear

data["MONTH_SIN"] = np.sin(
    2 * np.pi * data["MONTH"] / 12
)

data["MONTH_COS"] = np.cos(
    2 * np.pi * data["MONTH"] / 12
)

data["DAY_SIN"] = np.sin(
    2 * np.pi * data["DAY_OF_YEAR"] / 365.25
)

data["DAY_COS"] = np.cos(
    2 * np.pi * data["DAY_OF_YEAR"] / 365.25
)

# 4. Define features and target
# Do NOT include rainfall as an input feature.

features = [
    "TEMPERATURE",
    "HUMIDITY",
    "WIND_SPEED",
    "PRESSURE",
    "MONTH_SIN",
    "MONTH_COS",
    "DAY_SIN",
    "DAY_COS"
]

target = "RAINFALL"

X = data[features]
y = data[target]

# 5. Chronological data splitting
train_mask = data["DATE"].dt.year <= 2023

validation_mask = (
    (data["DATE"].dt.year >= 2024) &
    (data["DATE"].dt.year <= 2025)
)

test_mask = data["DATE"].dt.year == 2026

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[validation_mask], y[validation_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("\nDATASET INFORMATION")
print("-" * 65)
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))
print("Input features:", len(features))

# 6. Train two candidate models
experiments = {
    "Gradient Boosting Log (Absolute Error)": TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(
            n_estimators=300,        # Increased from 200 to learn more patterns
            learning_rate=0.05,
            max_depth=5,             # Increased from 3 to capture complex storms
            min_samples_leaf=5,
            loss="absolute_error",   # Changed from squared_error
            random_state=42
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ),

    "Gradient Boosting Log (Huber)": TransformedTargetRegressor(
        regressor=GradientBoostingRegressor(
            n_estimators=300,        
            learning_rate=0.05,
            max_depth=5,             
            min_samples_leaf=5,
            loss="huber",            # Huber is great for handling outliers
            random_state=42
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )
}

results = []
trained_models = {}

print("\nTRAINING MODELS")
print("-" * 65)

for name, model in experiments.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = np.maximum(
        model.predict(X_val), 0
    )

    mae = mean_absolute_error(y_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

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

# 7. Select the best model using validation MAE
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Validation_MAE"
)

comparison.to_csv(
    "forecast_model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("-" * 65)
print(comparison.to_string(index=False))

best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nSELECTED MODEL:", best_name)

# 8. Evaluate on the untouched test period
test_predictions = np.maximum(
    best_model.predict(X_test), 0
)

test_mae = mean_absolute_error(
    y_test, test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

test_r2 = r2_score(
    y_test, test_predictions
)

print("\nFINAL TEST RESULTS")
print("-" * 65)
print(f"Test MAE: {test_mae:.4f} mm")
print(f"Test RMSE: {test_rmse:.4f} mm")
print(f"Test R2: {test_r2:.4f}")

# 9. Save the model and its feature information
joblib.dump(
    {
        "model": best_model,
        "features": features,
        "model_name": best_name,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "wind_speed_unit": "m/s",
        "pressure_unit": "kPa"
    },
    "forecast_rainfall_model.pkl"
)

# 10. Save test predictions
output = pd.DataFrame({
    "DATE": data.loc[test_mask, "DATE"].values,
    "ACTUAL_RAINFALL": y_test.values,
    "PREDICTED_RAINFALL": test_predictions
})

output.to_csv(
    "forecast_model_predictions.csv",
    index=False
)

print("\nTRAINING COMPLETED!")
print("Generated files:")
print("- forecast_model_comparison.csv")
print("- forecast_model_predictions.csv")
print("- forecast_rainfall_model.pkl")