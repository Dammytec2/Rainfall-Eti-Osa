
import requests
import pandas as pd
import os
from io import StringIO
from datetime import date, timedelta

# =====================================
# 1. LOCATION SETTINGS
# =====================================

latitude = 6.4525
longitude = 3.5365

start_date = "20000101"

# Allow 3 days for NASA POWER data publication
latest_date = date.today() - timedelta(days=3)
end_date = latest_date.strftime("%Y%m%d")

base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

parameters = {
    "PRECTOTCORR": "rainfall",
    "T2M": "temperature",
    "RH2M": "humidity",
    "WS2M": "wind_speed",
    "PS": "pressure"
}

# =====================================
# 2. DOWNLOAD WEATHER DATA
# =====================================

print("NASA POWER Weather Data - Eti-Osa, Lagos")
print("=" * 55)

print(f"Start date: {start_date}")
print(f"Requested end date: {end_date}")

successful = 0
failed = 0

for param_code, param_name in parameters.items():

    print(f"\nDownloading {param_name}...")

    params = {
        "parameters": param_code,
        "community": "AG",
        "latitude": latitude,
        "longitude": longitude,
        "start": start_date,
        "end": end_date,
        "format": "CSV"
    }

    try:
        response = requests.get(
            base_url,
            params=params,
            timeout=180
        )

        response.raise_for_status()

        lines = response.text.splitlines()

        # Locate actual CSV header
        header_row = next(
            (
                i for i, line in enumerate(lines)
                if "YEAR" in line and "DOY" in line
            ),
            None
        )

        if header_row is None:
            raise ValueError("CSV header not found in NASA response.")

        csv_data = "\n".join(lines[header_row:])

        df = pd.read_csv(StringIO(csv_data))

        df.columns = df.columns.str.strip()

        # Validate returned data
        required_columns = ["YEAR", "DOY", param_code]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Expected column {column} was not returned."
                )

        if df.empty:
            raise ValueError("NASA returned an empty dataset.")

        # Check actual date range returned
        df["DATE"] = (
            pd.to_datetime(
                df["YEAR"].astype(str),
                format="%Y"
            )
            + pd.to_timedelta(df["DOY"] - 1, unit="D")
        )

        actual_start = df["DATE"].min().date()
        actual_end = df["DATE"].max().date()

        # Remove temporary DATE column to preserve original format
        df = df.drop(columns=["DATE"])

        # Save to temporary file first
        filename = f"{param_name}.csv"
        temp_filename = f"{param_name}_temp.csv"

        df.to_csv(temp_filename, index=False)

        # Replace old file only after successful download and validation
        os.replace(temp_filename, filename)

        print(f"SUCCESS: {filename}")
        print(f"Records: {len(df)}")
        print(f"Actual data range: {actual_start} to {actual_end}")

        if actual_end < latest_date:
            print(
                "NOTICE: NASA has not yet returned data "
                "up to the requested ending date."
            )

        successful += 1

    except Exception as e:
        print(f"FAILED: {param_name}")
        print(f"Reason: {e}")
        failed += 1

# =====================================
# 3. DOWNLOAD SUMMARY
# =====================================

print("\n" + "=" * 55)
print("FINAL DOWNLOAD REPORT")
print("=" * 55)

print(f"Successful datasets: {successful}")
print(f"Failed datasets: {failed}")

if successful == 5:
    print("\nAll five datasets downloaded successfully!")
    print(f"Saved in: {os.getcwd()}")

elif successful > 0:
    print("\nSome datasets downloaded successfully.")

else:
    print("\nNo datasets were successfully processed.")
