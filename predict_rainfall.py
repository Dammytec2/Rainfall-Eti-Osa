
import requests
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

MODEL_FILE = "forecast_rainfall_model.pkl"

saved_model = joblib.load(MODEL_FILE)

model = saved_model["model"]
features = saved_model["features"]
model_name = saved_model["model_name"]

print("=" * 65)
print("ETI-OSA RAINFALL PREDICTION SYSTEM")
print("=" * 65)
print(f"Model loaded: {model_name}")

# ==========================================
# 2. DOWNLOAD 16-DAY WEATHER FORECAST
# ==========================================

LATITUDE = 6.4525
LONGITUDE = 3.5365

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "daily": [
        "precipitation_sum",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean"
    ],
    "hourly": "surface_pressure",
    "forecast_days": 16,
    "timezone": "Africa/Lagos"
}

print("\nDownloading weather forecast...")

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

weather_data = response.json()

daily = pd.DataFrame(weather_data["daily"])

# ==========================================
# 3. PREPARE WEATHER FEATURES
# ==========================================

forecast = pd.DataFrame()

forecast["DATE"] = pd.to_datetime(daily["time"])

forecast["TEMPERATURE"] = daily["temperature_2m_mean"]
forecast["HUMIDITY"] = daily["relative_humidity_2m_mean"]

# Convert wind speed from km/h to m/s
forecast["WIND_SPEED"] = (
    daily["wind_speed_10m_mean"] / 3.6
)

# Convert hourly pressure from hPa to kPa
hourly = pd.DataFrame(weather_data["hourly"])

hourly["DATE"] = pd.to_datetime(
    hourly["time"]
).dt.date

hourly["PRESSURE_KPA"] = (
    hourly["surface_pressure"] / 10
)

daily_pressure = hourly.groupby(
    "DATE"
)["PRESSURE_KPA"].mean()

forecast["PRESSURE"] = forecast["DATE"].dt.date.map(
    daily_pressure
)

# Keep the API rainfall forecast separate.
# It is NOT used as an input to our machine-learning model.
forecast["OPEN_METEO_RAINFALL"] = daily["precipitation_sum"]

# ==========================================
# 4. CREATE SEASONAL FEATURES
# ==========================================

forecast["MONTH"] = forecast["DATE"].dt.month
forecast["DAY_OF_YEAR"] = forecast["DATE"].dt.day_of_year

forecast["MONTH_SIN"] = np.sin(
    2 * np.pi * forecast["MONTH"] / 12
)

forecast["MONTH_COS"] = np.cos(
    2 * np.pi * forecast["MONTH"] / 12
)

forecast["DAY_SIN"] = np.sin(
    2 * np.pi * forecast["DAY_OF_YEAR"] / 365.25
)

forecast["DAY_COS"] = np.cos(
    2 * np.pi * forecast["DAY_OF_YEAR"] / 365.25
)

# ==========================================
# 5. MAKE MODEL PREDICTIONS
# ==========================================

X_forecast = forecast[features]

predictions = model.predict(X_forecast)

# Rainfall cannot be negative
forecast["PREDICTED_RAINFALL"] = np.maximum(
    predictions, 0
)

# ==========================================
# 6. CLASSIFY RAINFALL
# ==========================================

def rainfall_category(amount):

    if amount == 0:
        return "No rain"

    elif amount <= 2.5:
        return "Light rain"

    elif amount <= 10:
        return "Moderate rain"

    elif amount <= 50:
        return "Heavy rain"

    else:
        return "Very heavy rain"


forecast["RAIN_CATEGORY"] = (
    forecast["PREDICTED_RAINFALL"].apply(
        rainfall_category
    )
)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n16-DAY RAINFALL PREDICTIONS")
print("=" * 65)

for _, row in forecast.iterrows():

    print(f"\nDate: {row['DATE'].strftime('%d %B %Y')}")

    print(
        f"Predicted rainfall: "
        f"{row['PREDICTED_RAINFALL']:.2f} mm"
    )

    print(f"Rain category: {row['RAIN_CATEGORY']}")

    print(
        f"Temperature: {row['TEMPERATURE']:.1f} °C"
    )

    print(
        f"Humidity: {row['HUMIDITY']:.1f}%"
    )

    print(
        f"Wind speed: {row['WIND_SPEED']:.2f} m/s"
    )

    print(
        f"Pressure: {row['PRESSURE']:.2f} kPa"
    )

    print(
        f"Open-Meteo rainfall reference: "
        f"{row['OPEN_METEO_RAINFALL']:.2f} mm"
    )

# ==========================================
# 8. SAVE PREDICTIONS
# ==========================================

output_columns = [
    "DATE",
    "TEMPERATURE",
    "HUMIDITY",
    "WIND_SPEED",
    "PRESSURE",
    "PREDICTED_RAINFALL",
    "RAIN_CATEGORY",
    "OPEN_METEO_RAINFALL"
]

forecast[output_columns].to_csv(
    "eti_osa_ml_predictions.csv",
    index=False
)

print("\n" + "=" * 65)
print("PREDICTIONS COMPLETED SUCCESSFULLY!")
print("Saved as: eti_osa_ml_predictions.csv")
print("=" * 65)
