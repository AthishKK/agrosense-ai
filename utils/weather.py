# ================================================
# AgroSense AI — OpenWeatherMap Integration
# utils/weather.py
# ================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import streamlit as st


# ─────────────────────────────────────────────────
# MAIN FUNCTION — cached for 10 minutes (600 s)
# ─────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_weather(city_name: str, api_key: str) -> dict:
    """
    Fetch current weather for a city from OpenWeatherMap.

    Parameters
    ----------
    city_name : str   e.g. "Mumbai", "Delhi", "Chennai"
    api_key   : str   OWM API key from secrets.toml

    Returns
    -------
    dict with keys:
        success     : bool
        temperature : float  (°C)
        humidity    : float  (%)
        rainfall    : float  (mm — last 1h or 3h, 0.0 if dry)
        description : str    e.g. "light rain"
        icon_code   : str    e.g. "10d"  (use in OWM icon URL)
        icon_url    : str    full URL to weather icon PNG
        city        : str    resolved city name from API
        country     : str    2-letter country code
        feels_like  : float  (°C)
        wind_speed  : float  (m/s)
        error       : str    empty string on success, message on failure
    """

    # ── Validate inputs ──
    if not city_name or not city_name.strip():
        return _error_result("City name cannot be empty.")

    if not api_key or not api_key.strip():
        return _error_result("WEATHER_API_KEY is not set in .streamlit/secrets.toml.")

    # ── Call OWM Current Weather API ──
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name.strip()}"
        f"&appid={api_key.strip()}"
        "&units=metric"
    )

    try:
        response = requests.get(url, timeout=8)
        data = response.json()

        # ── Handle API-level errors ──
        if response.status_code == 401:
            return _error_result("Invalid API key. Check WEATHER_API_KEY in secrets.toml.")
        if response.status_code == 404:
            return _error_result(f"City '{city_name}' not found. Try a different spelling.")
        if response.status_code == 429:
            return _error_result("API rate limit reached. Please wait a minute and try again.")
        if response.status_code != 200:
            msg = data.get("message", f"API error (HTTP {response.status_code})")
            return _error_result(msg)

        # ── Parse response ──
        main    = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind    = data.get("wind", {})
        rain    = data.get("rain", {})

        temperature = round(main.get("temp", 0.0), 1)
        humidity    = round(main.get("humidity", 0.0), 1)
        feels_like  = round(main.get("feels_like", 0.0), 1)
        wind_speed  = round(wind.get("speed", 0.0), 1)
        description = weather.get("description", "").capitalize()
        icon_code   = weather.get("icon", "01d")

        # Rainfall: prefer 1h, fall back to 3h, default 0.0
        rainfall = round(
            rain.get("1h", rain.get("3h", 0.0)), 2
        )

        icon_url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"

        return {
            "success":     True,
            "temperature": temperature,
            "humidity":    humidity,
            "rainfall":    rainfall,
            "description": description,
            "icon_code":   icon_code,
            "icon_url":    icon_url,
            "city":        data.get("name", city_name.title()),
            "country":     data.get("sys", {}).get("country", ""),
            "feels_like":  feels_like,
            "wind_speed":  wind_speed,
            "error":       "",
        }

    except requests.exceptions.ConnectionError:
        return _error_result("No internet connection. Please check your network.")
    except requests.exceptions.Timeout:
        return _error_result("Request timed out. OpenWeatherMap did not respond in 8 seconds.")
    except Exception as e:
        return _error_result(f"Unexpected error: {str(e)}")


# ─────────────────────────────────────────────────
# HELPER — build a failed result dict
# ─────────────────────────────────────────────────

def _error_result(message: str) -> dict:
    return {
        "success":     False,
        "temperature": 0.0,
        "humidity":    0.0,
        "rainfall":    0.0,
        "description": "",
        "icon_code":   "",
        "icon_url":    "",
        "city":        "",
        "country":     "",
        "feels_like":  0.0,
        "wind_speed":  0.0,
        "error":       message,
    }


# ─────────────────────────────────────────────────
# QUICK TEST (run directly: python utils/weather.py)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_KEY  = "8976180116e4d9a521174dbdd87c1d89"
    TEST_CITY = "Mumbai"

    print("=" * 50)
    print(f"Testing get_weather('{TEST_CITY}')")
    print("=" * 50)

    result = get_weather(TEST_CITY, TEST_KEY)

    if result["success"]:
        print(f"✅ City        : {result['city']}, {result['country']}")
        print(f"   Temperature : {result['temperature']}°C  (feels like {result['feels_like']}°C)")
        print(f"   Humidity    : {result['humidity']}%")
        print(f"   Rainfall    : {result['rainfall']} mm")
        print(f"   Description : {result['description']}")
        print(f"   Wind Speed  : {result['wind_speed']} m/s")
        print(f"   Icon Code   : {result['icon_code']}")
        print(f"   Icon URL    : {result['icon_url']}")
    else:
        print(f"❌ Error: {result['error']}")

    print("=" * 50)
