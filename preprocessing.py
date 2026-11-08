
import pandas as pd
import numpy as np

print("RAIN FALL PREDICTION - DATA PREPROCESSING")
print("=" * 50)

# 1. Load the master dataset
df = pd.read_csv("eti_osa_weather_master.csv")

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

print("\nOriginal dataset shape:", df.shape)

# 2. Check data quality
print("\nChecking data quality...")

print("Duplicate dates:", df["DATE"].duplicated().sum())
print("Missing values:\n", df.isnull().sum())

if df["DATE"].duplicated().any():
    raise ValueError("Duplicate dates found. Please investigate them.")

if df["RAINFALL"].lt(0).any():
    raise ValueError("Negative rainfall values found. Please investigate them.")

# Check whether any calendar dates are missing
expected_dates = pd.date_range(
    start=df["DATE"].min(),
    end=df["DATE"].max(),
    freq="D"
)

missing_dates = expected_dates.difference(df["DATE"])

print("Missing calendar dates:", len(missing_dates))

if len(missing_dates) > 0:
    raise ValueError(
        "Missing calendar dates found. "
        "Please investigate before creating lag features."
    )

if df.isnull().any().any():
    raise ValueError("Missing values found. Please investigate before continuing.")

# 3. Create the next-day rainfall target
df["TARGET_DATE"] = df["DATE"].shift(-1)
df["TARGET_RAINFALL"] = df["RAINFALL"].shift(-1)

# 4. Create calendar features for the prediction date
df["TARGET_MONTH"] = df["TARGET_DATE"].dt.month
df["TARGET_DAY_OF_YEAR"] = df["TARGET_DATE"].dt.day_of_year

df["MONTH_SIN"] = np.sin(
    2 * np.pi * df["TARGET_MONTH"] / 12
)

df["MONTH_COS"] = np.cos(
    2 * np.pi * df["TARGET_MONTH"] / 12
)

df["DAY_SIN"] = np.sin(
    2 * np.pi * df["TARGET_DAY_OF_YEAR"] / 365.25
)

df["DAY_COS"] = np.cos(
    2 * np.pi * df["TARGET_DAY_OF_YEAR"] / 365.25
)

# 5. Create weather lag features
weather_columns = [
    "RAINFALL",
    "TEMPERATURE",
    "HUMIDITY",
    "WIND_SPEED",
    "PRESSURE"
]

# Current day's weather is available when forecasting tomorrow
for column in weather_columns:
    df[f"{column}_TODAY"] = df[column]

# Previous weather observations
for column in weather_columns:
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"{column}_LAG_{lag}"] = df[column].shift(lag)

# 6. Create previous rainfall averages
for window in [7, 14, 30]:
    df[f"RAINFALL_ROLLING_MEAN_{window}"] = (
        df["RAINFALL"].rolling(window=window).mean()
    )

# 7. Remove rows without complete features or target
df = df.dropna().reset_index(drop=True)

print("\nProcessed dataset shape:", df.shape)

# 8. Select model features and target
excluded_columns = [
    "DATE",
    "YEAR",
    "DOY",
    "TARGET_DATE",
    "TARGET_RAINFALL",
    "MONTH_NAME",
    "RAINFALL",
    "TEMPERATURE",
    "HUMIDITY",
    "WIND_SPEED",
    "PRESSURE"
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

X = df[feature_columns]
y = df["TARGET_RAINFALL"]

# 9. Split data chronologically

train_mask = df["TARGET_DATE"].dt.year <= 2023

validation_mask = (
    (df["TARGET_DATE"].dt.year >= 2024) &
    (df["TARGET_DATE"].dt.year <= 2025)
)

test_mask = df["TARGET_DATE"].dt.year == 2026

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[validation_mask]
y_val = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# 10. Save the prepared datasets
train_data = df.loc[train_mask].copy()
validation_data = df.loc[validation_mask].copy()
test_data = df.loc[test_mask].copy()

train_data.to_csv("train_dataset.csv", index=False)
validation_data.to_csv("validation_dataset.csv", index=False)
test_data.to_csv("test_dataset.csv", index=False)

df.to_csv("model_ready_dataset.csv", index=False)

print("\nDATA PREPROCESSING COMPLETED!")
print("=" * 50)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nTraining period:",
      train_data["TARGET_DATE"].min(),
      "to",
      train_data["TARGET_DATE"].max())

print("\nValidation period:",
      validation_data["TARGET_DATE"].min(),
      "to",
      validation_data["TARGET_DATE"].max())

print("\nTesting period:",
      test_data["TARGET_DATE"].min(),
      "to",
      test_data["TARGET_DATE"].max())

print("\nNumber of input features:", len(feature_columns))

print("\nGenerated files:")
print("- model_ready_dataset.csv")
print("- train_dataset.csv")
print("- validation_dataset.csv")
print("- test_dataset.csv")
