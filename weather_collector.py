import requests
import pandas as pd
import os

print("🌤️ Collecting weather data for Vellore...")

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    'latitude': 12.9165,
    'longitude': 79.1325,
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'daily': 'temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum',
    'timezone': 'Asia/Kolkata'
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame({
    'datetime': pd.to_datetime(data['daily']['time']),
    'temperature': data['daily']['temperature_2m_mean'],
    'humidity': data['daily']['relative_humidity_2m_mean'],
    'wind_speed': data['daily']['wind_speed_10m_mean'],
    'weather': 'clear'
})

os.makedirs('data/raw/weather', exist_ok=True)
df.to_csv('data/raw/weather/vellore_weather_raw.csv', index=False)

print(f"✅ Collected {len(df)} days of weather data")
print(f"💾 Saved to: data/raw/weather/vellore_weather_raw.csv")
print(df.head())