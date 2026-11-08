import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Eti-Osa Rainfall Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background-color: #101d35;
    height: 100vh;
    overflow-y: auto;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    background: linear-gradient(120deg, #101d35, #123c59, #087e8b);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    color: #d8e9f0;
    font-size: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e8edf4;
    box-shadow: 0 4px 15px rgba(20,40,70,0.05);
    min-height: 135px;
}

.metric-label {
    color: #738198;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #17243b;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-unit {
    color: #8a96a8;
    font-size: 12px;
}

.prediction-panel {
    background: white;
    padding: 25px;
    border-radius: 17px;
    border: 1px solid #e8edf4;
    box-shadow: 0 4px 15px rgba(20,40,70,0.05);
}

.prediction-number {
    color: #087e8b;
    font-size: 42px;
    font-weight: 800;
}

.section-heading {
    color: #17243b;
    font-size: 22px;
    font-weight: 800;
    margin-top: 28px;
    margin-bottom: 12px;
}

.footer {
    text-align: center;
    color: #8490a3;
    padding: 30px 0;
    font-size: 12px;
}

.stDownloadButton button {
    background-color: #087e8b;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    saved_model = joblib.load("forecast_rainfall_model.pkl")

    return saved_model["model"], saved_model["features"]


model, features = load_model()


# =====================================================
# 4. RAINFALL CATEGORIES
# =====================================================

def rainfall_category(amount, weather_code):
    # Override for Thunderstorms (codes 95, 96, 99)
    if weather_code in [95, 96, 99]:
        return "Heavy rain"
        
    # Standard categorization based on ML amount
    if amount <= 0:
        return "No rain"
    elif amount <= 2.5:
        return "Light rain"
    elif amount <= 10:
        return "Moderate rain"
    elif amount <= 50:
        return "Heavy rain"
    else:
        return "Very heavy rain"


def category_style(category):

    styles = {
        "No rain": ("#e8edf4", "#526174"),
        "Light rain": ("#e1f4ff", "#0878ae"),
        "Moderate rain": ("#e0f7f4", "#087e72"),
        "Heavy rain": ("#fff0d9", "#b66a00"),
        "Very heavy rain": ("#ffe4e4", "#c03939")
    }

    return styles.get(category, ("#e8edf4", "#526174"))


# =====================================================
# 5. WEATHER DESCRIPTIONS
# =====================================================

def weather_description(code):

    descriptions = {
        0: ("☀️", "Clear sky"),
        1: ("🌤️", "Mainly clear"),
        2: ("⛅", "Partly cloudy"),
        3: ("☁️", "Overcast"),
        45: ("🌫️", "Fog"),
        48: ("🌫️", "Depositing rime fog"),
        51: ("🌦️", "Light drizzle"),
        53: ("🌦️", "Moderate drizzle"),
        55: ("🌧️", "Dense drizzle"),
        56: ("🌧️", "Freezing drizzle"),
        57: ("🌧️", "Dense freezing drizzle"),
        61: ("🌦️", "Slight rain"),
        63: ("🌧️", "Moderate rain"),
        65: ("🌧️", "Heavy rain"),
        66: ("🌧️", "Freezing rain"),
        67: ("🌧️", "Heavy freezing rain"),
        71: ("🌨️", "Slight snowfall"),
        73: ("🌨️", "Moderate snowfall"),
        75: ("❄️", "Heavy snowfall"),
        77: ("🌨️", "Snow grains"),
        80: ("🌦️", "Slight rain showers"),
        81: ("🌧️", "Moderate rain showers"),
        82: ("⛈️", "Violent rain showers"),
        85: ("🌨️", "Slight snow showers"),
        86: ("❄️", "Heavy snow showers"),
        95: ("⛈️", "Thunderstorm"),
        96: ("⛈️", "Thunderstorm with hail"),
        99: ("⛈️", "Heavy thunderstorm with hail")
    }

    try:
        code = int(code)
    except (TypeError, ValueError):
        return "🌥️", "Variable conditions"

    return descriptions.get(code, ("🌥️", "Variable conditions"))


# =====================================================
# 6. GET WEATHER FORECAST
# =====================================================

@st.cache_data(ttl=3600)
def get_weather_forecast():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 6.4525,
        "longitude": 3.5365,
        "daily": [
            "precipitation_sum",
            "precipitation_probability_max",
            "weather_code",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean"
        ],
        "hourly": "surface_pressure",
        "forecast_days": 16,
        "timezone": "Africa/Lagos"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    weather = response.json()

    daily = pd.DataFrame(weather["daily"])
    hourly = pd.DataFrame(weather["hourly"])

    forecast = pd.DataFrame()

    forecast["DATE"] = pd.to_datetime(daily["time"])
    forecast["TEMPERATURE"] = daily["temperature_2m_mean"]
    forecast["TEMP_MAX"] = daily["temperature_2m_max"]
    forecast["TEMP_MIN"] = daily["temperature_2m_min"]
    forecast["HUMIDITY"] = daily["relative_humidity_2m_mean"]
    forecast["RAIN_PROBABILITY"] = daily["precipitation_probability_max"]
    forecast["WEATHER_CODE"] = daily["weather_code"]

    # Convert wind speed from km/h to m/s
    forecast["WIND_SPEED"] = daily["wind_speed_10m_mean"] / 3.6

    # Convert pressure from hPa to kPa
    hourly["DATE"] = pd.to_datetime(hourly["time"]).dt.date
    hourly["PRESSURE"] = hourly["surface_pressure"] / 10

    daily_pressure = hourly.groupby("DATE")["PRESSURE"].mean()

    forecast["PRESSURE"] = forecast["DATE"].dt.date.map(daily_pressure)

    # Open-Meteo rainfall is for comparison only
    forecast["OPEN_METEO_RAINFALL"] = daily["precipitation_sum"]

    # Calendar features
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

    # Machine learning prediction
    predictions = model.predict(forecast[features].copy())

    forecast["PREDICTED_RAINFALL"] = np.maximum(predictions, 0)

    # Apply the new category function using both the ML amount and the Open-Meteo weather code
    forecast["RAIN_CATEGORY"] = forecast.apply(
        lambda row: rainfall_category(row["PREDICTED_RAINFALL"], row["WEATHER_CODE"]), 
        axis=1
    )

    return forecast


# =====================================================
# 7. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🌧️ RAINFALL")
    st.subheader("INTELLIGENCE SYSTEM")

    st.divider()

    st.subheader("📍 Monitoring Location")
    st.write("**Eti-Osa, Lagos State**")
    st.caption("Nigeria | 6.4525°N, 3.5365°E")

    st.divider()

    st.subheader("🤖 Model Information")
    st.write("Gradient Boosting Log")
    st.caption("Machine Learning Regression Model")

    st.divider()

    st.subheader("📊 Rainfall Categories")
    st.write("🔵 Light: Above 0–2.5 mm")
    st.write("🟢 Moderate: Above 2.5–10 mm")
    st.write("🟠 Heavy: Above 10–50 mm")
    st.write("🔴 Very Heavy: Above 50 mm")

    st.divider()

    st.caption("Eti-Osa Rainfall Prediction Project")


# =====================================================
# 8. MAIN DASHBOARD
# =====================================================

st.markdown("""
<div class="hero">
    <span style="background:rgba(255,255,255,0.15);padding:7px 13px;border-radius:20px;">
        SMART WEATHER MONITORING
    </span>
    <h1>🌧️ Eti-Osa Rainfall Intelligence</h1>
    <p>
        Machine-learning-powered rainfall prediction
        and weather monitoring for Eti-Osa, Lagos State.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 9. DISPLAY FORECAST
# =====================================================

try:

    with st.spinner("Loading the latest weather forecast..."):
        forecast = get_weather_forecast()

    st.success("Live weather data connected")

    available_dates = forecast["DATE"].dt.date.tolist()

    selected_date = st.sidebar.selectbox(
        "📅 Select Forecast Date",
        available_dates,
        format_func=lambda d: d.strftime("%d %B %Y")
    )

    selected = forecast[
        forecast["DATE"].dt.date == selected_date
    ].iloc[0]

    weather_icon, weather_text = weather_description(
        selected["WEATHER_CODE"]
    )

    # SELECTED DATE PREDICTION

    st.markdown('<div class="section-heading">Selected Date Prediction</div>', unsafe_allow_html=True)

    bg, fg = category_style(selected["RAIN_CATEGORY"])

    st.markdown(f"""
    <div class="prediction-panel">
        <div style="color:#77849a;font-size:13px;">
            FORECAST FOR {selected_date.strftime('%A, %d %B %Y').upper()}
        </div>
        <div class="prediction-number">
            {selected['PREDICTED_RAINFALL']:.2f} mm
        </div>
        <span style="background:{bg};color:{fg};padding:7px 13px;border-radius:20px;font-size:12px;font-weight:700;">
            {selected['RAIN_CATEGORY']}
        </span>
        <div style="margin-top:15px;font-size:18px;color:#17243b;font-weight:600;">
            {weather_icon} {weather_text}
        </div>
        <p style="color:#8490a3;font-size:13px;margin-top:15px;">
            Machine learning rainfall prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # WEATHER OVERVIEW

    st.markdown('<div class="section-heading">Weather Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "🌡️", "Temperature", f"{selected['TEMPERATURE']:.1f}", "°C"),
        (col2, "💧", "Humidity", f"{selected['HUMIDITY']:.1f}", "%"),
        (
            col3,
            "🌧️",
            "Rain Probability",
            f"{selected['RAIN_PROBABILITY']:.0f}" if pd.notna(selected["RAIN_PROBABILITY"]) else "N/A",
            "%"
        ),
        (col4, "💨", "Wind Speed", f"{selected['WIND_SPEED']:.2f}", "m/s")
    ]

    for col, icon, label, value, unit in metrics:

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        f"Temperature range: {selected['TEMP_MIN']:.1f}°C – "
        f"{selected['TEMP_MAX']:.1f}°C"
    )

    st.caption(
        "Rain probability comes from Open-Meteo. "
        "It is separate from the machine-learning rainfall prediction."
    )

    # ADDITIONAL WEATHER DETAILS

    st.markdown('<div class="section-heading">Additional Weather Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:24px;">🌍</div>
            <div class="metric-label">Atmospheric Pressure</div>
            <div class="metric-value">{selected['PRESSURE']:.2f}</div>
            <div class="metric-unit">kPa</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:24px;">🌧️</div>
            <div class="metric-label">Open-Meteo Rainfall Reference</div>
            <div class="metric-value">{selected['OPEN_METEO_RAINFALL']:.2f}</div>
            <div class="metric-unit">mm</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Open-Meteo rainfall is displayed for comparison only. "
        "It is not used as an input to the machine-learning model."
    )

    # 16-DAY SUMMARY

    st.markdown('<div class="section-heading">16-Day Forecast Summary</div>', unsafe_allow_html=True)

    total_rain = forecast["PREDICTED_RAINFALL"].sum()
    average_rain = forecast["PREDICTED_RAINFALL"].mean()
    rainy_days = (forecast["PREDICTED_RAINFALL"] > 0).sum()

    col1, col2, col3 = st.columns(3)

    summary_metrics = [
        (col1, "🌧️", "Total Predicted Rainfall", f"{total_rain:.2f} mm"),
        (col2, "📈", "Average Daily Rainfall", f"{average_rain:.2f} mm"),
        (col3, "📅", "Days with Predicted Rain", f"{rainy_days} of 16 days")
    ]

    for col, icon, label, value in summary_metrics:

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:24px;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # RAINFALL CHART

    st.markdown('<div class="section-heading">Rainfall Prediction Trend</div>', unsafe_allow_html=True)

    st.caption("Daily machine-learning rainfall predictions in millimetres.")

    chart_data = forecast.set_index("DATE")[["PREDICTED_RAINFALL"]]

    st.area_chart(
        chart_data,
        color="#087e8b",
        use_container_width=True
    )

    # DETAILED FORECAST TABLE

    st.markdown('<div class="section-heading">Detailed 16-Day Forecast</div>', unsafe_allow_html=True)

    display_data = forecast[
        [
            "DATE",
            "WEATHER_CODE",
            "PREDICTED_RAINFALL",
            "RAIN_CATEGORY",
            "RAIN_PROBABILITY",
            "TEMPERATURE",
            "TEMP_MIN",
            "TEMP_MAX",
            "HUMIDITY"
        ]
    ].copy()

    display_data["WEATHER"] = display_data["WEATHER_CODE"].apply(
        lambda code: weather_description(code)[0] + " " + weather_description(code)[1]
    )

    display_data["DATE"] = display_data["DATE"].dt.strftime("%d %b %Y")

    display_data = display_data.rename(
        columns={
            "DATE": "Date",
            "PREDICTED_RAINFALL": "Rainfall (mm)",
            "RAIN_CATEGORY": "Rainfall Category",
            "RAIN_PROBABILITY": "Rain Probability (%)",
            "TEMPERATURE": "Temperature (°C)",
            "TEMP_MIN": "Min Temp (°C)",
            "TEMP_MAX": "Max Temp (°C)",
            "HUMIDITY": "Humidity (%)"
        }
    )

    display_data = display_data.drop(columns=["WEATHER_CODE"])

    display_data = display_data[
        [
            "Date",
            "WEATHER",
            "Rainfall (mm)",
            "Rainfall Category",
            "Rain Probability (%)",
            "Temperature (°C)",
            "Min Temp (°C)",
            "Max Temp (°C)",
            "Humidity (%)"
        ]
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    # DOWNLOAD REPORT

    st.markdown('<div class="section-heading">Download Forecast Report</div>', unsafe_allow_html=True)

    csv_data = forecast.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast Results (CSV)",
        data=csv_data,
        file_name="eti_osa_rainfall_predictions.csv",
        mime="text/csv"
    )

    # MODEL COMPARISON

    with st.expander("Compare ML Model with Open-Meteo Forecast"):

        comparison = forecast[
            ["DATE", "PREDICTED_RAINFALL", "OPEN_METEO_RAINFALL"]
        ].copy()

        comparison["DATE"] = comparison["DATE"].dt.strftime("%d %b %Y")

        comparison = comparison.rename(
            columns={
                "DATE": "Date",
                "PREDICTED_RAINFALL": "Our Model (mm)",
                "OPEN_METEO_RAINFALL": "Open-Meteo (mm)"
            }
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are independent rainfall forecasts. "
            "Open-Meteo rainfall is not an input to our model."
        )

    # FOOTER

    st.markdown("""
    <div class="footer">
        <b>ETI-OSA RAINFALL INTELLIGENCE SYSTEM</b>
        <p>Machine Learning | Weather Data | Rainfall Prediction</p>
        <p>Developed for Eti-Osa, Lagos State, Nigeria.</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 10. ERROR HANDLING
# =====================================================

except Exception as error:

    st.error(f"Unable to load the rainfall forecast: {error}")

    st.info(
        "Check your internet connection and confirm that "
        "forecast_rainfall_model.pkl is in your project folder."
    )