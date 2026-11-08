
import pandas as pd
import os

print("MERGING ETI-OSA WEATHER DATA")
print("=" * 50)

# Load the five downloaded datasets
rainfall = pd.read_csv("rainfall.csv")
temperature = pd.read_csv("temperature.csv")
humidity = pd.read_csv("humidity.csv")
wind_speed = pd.read_csv("wind_speed.csv")
pressure = pd.read_csv("pressure.csv")

# Merge datasets using YEAR and DOY
df = rainfall.merge(
    temperature, on=["YEAR", "DOY"]
)

df = df.merge(
    humidity, on=["YEAR", "DOY"]
)

df = df.merge(
    wind_speed, on=["YEAR", "DOY"]
)

df = df.merge(
    pressure, on=["YEAR", "DOY"]
)

# Create a proper date column
df["DATE"] = pd.to_datetime(
    df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
    format="%Y%j"
)

# Rename columns
df = df.rename(columns={
    "PRECTOTCORR": "RAINFALL",
    "T2M": "TEMPERATURE",
    "RH2M": "HUMIDITY",
    "WS2M": "WIND_SPEED",
    "PS": "PRESSURE"
})

# Arrange columns
df = df[
    [
        "DATE",
        "YEAR",
        "DOY",
        "RAINFALL",
        "TEMPERATURE",
        "HUMIDITY",
        "WIND_SPEED",
        "PRESSURE"
    ]
]

# Sort by date
df = df.sort_values("DATE").reset_index(drop=True)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Save master dataset
df.to_csv("eti_osa_weather_master.csv", index=False)

# Display results
print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Start date: {df['DATE'].min()}")
print(f"End date: {df['DATE'].max()}")

print("\nSUCCESS! Master dataset created.")
print("File: eti_osa_weather_master.csv")
