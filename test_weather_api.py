
import requests
import pandas as pd

# Eti-Osa, Lagos coordinates
LATITUDE = 6.4525
LONGITUDE = 3.5365

URL = "https://api.open-meteo.com/v1/forecast"

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

print("FETCHING 16-DAY WEATHER FORECAST")
print("=" * 50)

try:
    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    daily = data["daily"]

    weather = pd.DataFrame({
        "DATE": daily["time"],
        "RAINFALL": daily["precipitation_sum"],
        "TEMPERATURE": daily["temperature_2m_mean"],
        "HUMIDITY": daily["relative_humidity_2m_mean"],
        "WIND_SPEED": daily["wind_speed_10m_mean"]
    })

    # Calculate daily mean surface pressure from hourly values
    hourly = data["hourly"]
    pressure = pd.DataFrame({
        "DATE": pd.to_datetime(hourly["time"]).date,
        "PRESSURE": hourly["surface_pressure"]
    })

    pressure_daily = pressure.groupby("DATE")["PRESSURE"].mean()

    weather["PRESSURE"] = pd.to_datetime(weather["DATE"]).dt.date.map(
        pressure_daily
    )

    print("\nFORECAST SUCCESSFULLY RETRIEVED")
    print("=" * 50)
    print(weather.to_string(index=False))

    weather.to_csv("eti_osa_16_day_forecast.csv", index=False)

    print("\nSaved: eti_osa_16_day_forecast.csv")
    print("Number of forecast days:", len(weather))

except requests.exceptions.RequestException as error:
    print("API CONNECTION ERROR:", error)

except (KeyError, TypeError, ValueError) as error:
    print("DATA PROCESSING ERROR:", error)
