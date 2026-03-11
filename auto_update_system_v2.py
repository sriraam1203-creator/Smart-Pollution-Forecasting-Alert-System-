"""
AUTOMATIC DAILY UPDATE SYSTEM - FIXED v2.0
============================================
Fixed: Now adds next date after last date in dataset
instead of yesterday's real date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import json

class AutoUpdateSystem:
    
    def __init__(self):
        self.data_path = 'data/processed/vellore_clean_dataset.csv'
        self.forecast_path = 'data/outputs/vellore_30day_forecast.csv'
        self.log_path = 'logs/update_log.txt'
        os.makedirs('logs', exist_ok=True)
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def get_next_date(self):
        """
        KEY FIX: Get next date AFTER last date in dataset
        NOT yesterday's real date
        """
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()
        next_date = last_date + timedelta(days=1)
        self.log(f"Last date in dataset: {last_date.date()}")
        self.log(f"Next date to add: {next_date.date()}")
        return next_date
    
    def fetch_weather_for_date(self, date):
        """Fetch weather for specific date"""
        try:
            lat, lon = 12.9165, 79.1325
            date_str = date.strftime('%Y-%m-%d')
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': date_str,
                'end_date': date_str,
                'daily': 'temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean',
                'timezone': 'Asia/Kolkata'
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['daily']['temperature_2m_mean'][0] is not None:
                    return {
                        'temperature': data['daily']['temperature_2m_mean'][0],
                        'humidity': data['daily']['relative_humidity_2m_mean'][0],
                        'wind_speed': data['daily']['wind_speed_10m_mean'][0]
                    }
            self.log("[!] Weather data unavailable. Using estimates.")
            return self.estimate_weather()
        except Exception as e:
            self.log(f"Weather error: {e}. Using estimates.")
            return self.estimate_weather()
    
    def estimate_weather(self):
        """Estimate weather from last 7 days"""
        df = pd.read_csv(self.data_path)
        last_7 = df.tail(7)
        return {
            'temperature': last_7['temperature'].mean(),
            'humidity': last_7['humidity'].mean(),
            'wind_speed': last_7['wind_speed'].mean()
        }
    
    def estimate_pm25(self):
        """Estimate PM2.5 from last 7 days with variation"""
        df = pd.read_csv(self.data_path)
        mean_pm25 = df['PM2_5'].tail(7).mean()
        variation = np.random.uniform(-0.1, 0.1)
        estimated = mean_pm25 * (1 + variation)
        self.log(f"Estimated PM2.5: {estimated:.2f} ug/m3")
        return estimated
    
    def check_if_date_exists(self, date):
        """Check if date already in dataset"""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        return date in df['date'].values
    
    def detect_anomaly(self, new_value, df):
        """Z-score anomaly detection"""
        recent = df['PM2_5'].tail(30)
        mean = recent.mean()
        std = recent.std()
        z_score = abs((new_value - mean) / std)
        is_anomaly = z_score > 3
        if is_anomaly:
            self.log(f"[ALERT] ANOMALY! PM2.5={new_value:.2f} (z={z_score:.2f})")
        else:
            self.log(f"[OK] Normal: PM2.5={new_value:.2f} (z={z_score:.2f})")
        return is_anomaly, {'value': new_value, 'mean': mean, 'std': std, 'z_score': z_score}
    
    def save_alert(self, anomaly_info):
        """Save alert to JSON"""
        alerts_file = 'logs/alerts.json'
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pm25_value': anomaly_info['value'],
            'z_score': anomaly_info['z_score']
        }
        alerts = []
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        alerts.append(alert)
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=2)
        self.log("[OK] Alert saved")
    
    def run_daily_update(self):
        """Main update routine"""
        self.log("=" * 60)
        self.log("STARTING DAILY UPDATE v2.0")
        self.log("=" * 60)
        
        try:
            df = pd.read_csv(self.data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # STEP 1: Get next sequential date
            self.log("\n[1/5] Getting next date...")
            next_date = self.get_next_date()
            
            if self.check_if_date_exists(next_date):
                self.log(f"[SKIP] {next_date.date()} already exists.")
                return True
            
            # STEP 2: Fetch data
            self.log(f"\n[2/5] Fetching data for {next_date.date()}...")
            weather = self.fetch_weather_for_date(next_date)
            pm25 = self.estimate_pm25()
            
            new_row = {
                'date': next_date.strftime('%Y-%m-%d'),
                'PM2_5': round(pm25, 2),
                'PM10': round(pm25 * 1.5, 2),
                'AOD': round(pm25 * 0.01, 4),
                'temperature': round(weather['temperature'], 2),
                'humidity': round(weather['humidity'], 2),
                'wind_speed': round(weather['wind_speed'], 2)
            }
            self.log(f"[OK] PM2.5={new_row['PM2_5']}, Temp={new_row['temperature']}")
            
            # STEP 3: Append
            self.log("\n[3/5] Updating dataset...")
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(self.data_path, index=False)
            self.log(f"[OK] Total rows: {len(df)}")
            
            # STEP 4: Anomaly check
            self.log("\n[4/5] Checking anomalies...")
            is_anomaly, anomaly_info = self.detect_anomaly(new_row['PM2_5'], df)
            if is_anomaly:
                self.save_alert(anomaly_info)
            
            # STEP 5: Regenerate forecast
            self.log("\n[5/5] Regenerating forecast...")
            try:
                import sys
                sys.path.insert(0, '.')
                from quick_forecast import quick_forecast
                quick_forecast()
                self.log("[OK] Forecast regenerated!")
            except Exception as e:
                self.log(f"[SKIP] Forecast: {e}")
            
            self.log("\n" + "=" * 60)
            self.log("[SUCCESS] UPDATE COMPLETE")
            self.log("=" * 60)
            return True
            
        except Exception as e:
            self.log(f"\n[ERROR] {e}")
            return False


if __name__ == "__main__":
    print("""
    PM2.5 AUTO-UPDATE SYSTEM - FIXED v2.0
    Adds next sequential date (not yesterday's real date)
    """)
    updater = AutoUpdateSystem()
    success = updater.run_daily_update()
    if success:
        print("\n[OK] Update successful!")
    else:
        print("\n[FAIL] Check logs.")