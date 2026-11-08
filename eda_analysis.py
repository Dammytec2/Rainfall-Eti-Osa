
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD MASTER DATASET
# ==========================================

print("ETI-OSA RAINFALL DATA ANALYSIS")
print("=" * 55)

df = pd.read_csv("eti_osa_weather_master.csv")

df["DATE"] = pd.to_datetime(df["DATE"])

# Create additional date features
df["MONTH"] = df["DATE"].dt.month
df["MONTH_NAME"] = df["DATE"].dt.month_name()
df["YEAR"] = df["DATE"].dt.year

sns.set_theme(style="whitegrid")

# ==========================================
# 2. BASIC STATISTICAL ANALYSIS
# ==========================================

print("\nDATASET OVERVIEW")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df[[
    "RAINFALL",
    "TEMPERATURE",
    "HUMIDITY",
    "WIND_SPEED",
    "PRESSURE"
]].describe())

print("\nRAINFALL STATISTICS")
print("Average rainfall:", df["RAINFALL"].mean())
print("Maximum rainfall:", df["RAINFALL"].max())
print("Minimum rainfall:", df["RAINFALL"].min())
print("Total rainfall:", df["RAINFALL"].sum())

# ==========================================
# 3. YEARLY RAINFALL TREND
# ==========================================

yearly_rainfall = df.groupby("YEAR")["RAINFALL"].sum()

plt.figure(figsize=(12, 6))
yearly_rainfall.plot(kind="line", marker="o")

plt.title("Annual Rainfall Trend in Eti-Osa (2000-2025)")
plt.xlabel("Year")
plt.ylabel("Total Rainfall (mm)")
plt.tight_layout()
plt.savefig("annual_rainfall_trend.png", dpi=300)
plt.show()

# ==========================================
# 4. MONTHLY RAINFALL PATTERN
# ==========================================

monthly_rainfall = df.groupby("MONTH")["RAINFALL"].mean()

plt.figure(figsize=(12, 6))
monthly_rainfall.plot(kind="bar")

plt.title("Average Daily Rainfall by Month")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm/day)")
plt.xticks(
    range(12),
    [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    rotation=0
)
plt.tight_layout()
plt.savefig("monthly_rainfall_pattern.png", dpi=300)
plt.show()

# ==========================================
# 5. RAINFALL DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["RAINFALL"],
    bins=50,
    kde=True
)

plt.title("Rainfall Distribution in Eti-Osa")
plt.xlabel("Rainfall (mm/day)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("rainfall_distribution.png", dpi=300)
plt.show()

# ==========================================
# 6. WEATHER CORRELATION
# ==========================================

weather_columns = [
    "RAINFALL",
    "TEMPERATURE",
    "HUMIDITY",
    "WIND_SPEED",
    "PRESSURE"
]

correlation = df[weather_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Weather Variables Correlation - Eti-Osa")

plt.tight_layout()
plt.savefig("weather_correlation.png", dpi=300)
plt.show()

# ==========================================
# 7. MONTHLY RAINFALL BOX PLOT
# ==========================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="MONTH_NAME",
    y="RAINFALL",
    order=[
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]
)

plt.title("Monthly Rainfall Distribution")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm/day)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("monthly_rainfall_boxplot.png", dpi=300)
plt.show()

# ==========================================
# 8. SAVE ANALYSIS RESULTS
# ==========================================

yearly_rainfall.to_csv("yearly_rainfall_summary.csv")

monthly_rainfall.to_csv("monthly_rainfall_summary.csv")

print("\n" + "=" * 55)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 55)

print("\nGenerated files:")
print("1. annual_rainfall_trend.png")
print("2. monthly_rainfall_pattern.png")
print("3. rainfall_distribution.png")
print("4. weather_correlation.png")
print("5. monthly_rainfall_boxplot.png")
print("6. yearly_rainfall_summary.csv")
print("7. monthly_rainfall_summary.csv")
