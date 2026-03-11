"""
AUTOMATIC DAILY UPDATE SYSTEM - WINDOWS COMPATIBLE
===================================================
Fetches latest data, updates dataset, regenerates forecast
Fixed for Windows emoji encoding
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import joblib
import keras
import os
import json

class AutoUpdateSystem:
    
    def __init__(self):
        self.data_path = 'data/processed/vellore_clean_dataset.csv'
        self.forecast_path = 'data/outputs/vellore_30day_forecast.csv'
        self.model_path = 'models/simple_baseline_model.h5'
        self.scaler_path = 'models/simple_target_scaler.pkl'
        self.log_path = 'logs/update_log.txt'
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
    
    def log(self, message):
        """Log messages with timestamp - Windows compatible"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        # Write to log file with UTF-8 encoding for emoji support
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def fetch_latest_weather(self, date):
        """Fetch weather data from Open-Meteo API"""
        try:
            # Vellore coordinates
            lat, lon = 12.9165, 79.1325
            
            # Format date
            date_str = date.strftime('%Y-%m-%d')
            
            # API call
            url = f"https://archive-api.open-meteo.com/v1/archive"
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
                
                return {
                    'temperature': data['daily']['temperature_2m_mean'][0],
                    'humidity': data['daily']['relative_humidity_2m_mean'][0],
                    'wind_speed': data['daily']['wind_speed_10m_mean'][0]
                }
            else:
                self.log(f"Weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            self.log(f"Error fetching weather: {e}")
            return None
    
    def fetch_latest_cpcb(self, date):
        """
        Fetch CPCB data - PLACEHOLDER
        In real implementation, this would call CPCB API
        For now, we'll use interpolation from existing data
        """
        try:
            # Load existing data
            df = pd.read_csv(self.data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Get last 7 days average as estimate
            last_week = df.tail(7)
            
            # Estimate PM2.5 (in real scenario, fetch from API)
            estimated_pm25 = last_week['PM2_5'].mean()
            
            # Add some random variation (±10%)
            variation = np.random.uniform(-0.1, 0.1)
            estimated_pm25 *= (1 + variation)
            
            self.log(f"Estimated PM2.5: {estimated_pm25:.2f} ug/m3 (based on last 7-day average)")
            
            return {
                'PM2_5': estimated_pm25,
                'PM10': estimated_pm25 * 1.5,  # Typical PM10/PM2.5 ratio
                'AOD': estimated_pm25 * 0.01   # Rough correlation
            }
            
        except Exception as e:
            self.log(f"Error fetching CPCB: {e}")
            return None
    
    def fetch_latest_data(self):
        """Fetch all latest data for yesterday"""
        # Get yesterday's date (most recent complete day)
        yesterday = datetime.now() - timedelta(days=1)
        
        self.log(f"Fetching data for {yesterday.date()}...")
        
        # Fetch weather
        weather_data = self.fetch_latest_weather(yesterday)
        if not weather_data:
            self.log("[X] Failed to fetch weather data")
            return None
        
        # Fetch CPCB (or estimate)
        cpcb_data = self.fetch_latest_cpcb(yesterday)
        if not cpcb_data:
            self.log("[X] Failed to fetch CPCB data")
            return None
        
        # Combine
        new_row = {
            'date': yesterday.strftime('%Y-%m-%d'),
            'PM2_5': cpcb_data['PM2_5'],
            'PM10': cpcb_data['PM10'],
            'AOD': cpcb_data['AOD'],
            'temperature': weather_data['temperature'],
            'humidity': weather_data['humidity'],
            'wind_speed': weather_data['wind_speed']
        }
        
        self.log(f"[OK] Data fetched: PM2.5={new_row['PM2_5']:.2f}, Temp={new_row['temperature']:.1f}C")
        
        return new_row
    
    def check_if_data_exists(self, date):
        """Check if data for this date already exists"""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        return date in df['date'].values
    
    def append_new_data(self, new_row):
        """Append new data to dataset"""
        try:
            # Load existing dataset
            df = pd.read_csv(self.data_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Check if already exists
            new_date = pd.to_datetime(new_row['date'])
            if self.check_if_data_exists(new_date):
                self.log(f"[!] Data for {new_date.date()} already exists. Skipping append.")
                return df
            
            # Create new row DataFrame
            new_df = pd.DataFrame([new_row])
            
            # Append
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Save
            df.to_csv(self.data_path, index=False)
            
            self.log(f"[OK] Data appended. Dataset now has {len(df)} rows.")
            
            return df
            
        except Exception as e:
            self.log(f"[X] Error appending data: {e}")
            return None
    
    def detect_anomaly(self, new_value, df):
        """Detect if new PM2.5 value is anomalous"""
        # Get recent statistics (last 30 days)
        recent_values = df['PM2_5'].tail(30)
        
        mean = recent_values.mean()
        std = recent_values.std()
        
        # Check if beyond 3 standard deviations
        z_score = abs((new_value - mean) / std)
        
        is_anomaly = z_score > 3
        
        if is_anomaly:
            direction = "SPIKE" if new_value > mean else "DROP"
            self.log(f"[!!!] ANOMALY DETECTED! {direction}: PM2.5={new_value:.2f} (mean={mean:.2f}, std={std:.2f}, z={z_score:.2f})")
        else:
            self.log(f"[OK] Normal value: PM2.5={new_value:.2f} (within 3-sigma)")
        
        return is_anomaly, {
            'value': new_value,
            'mean': mean,
            'std': std,
            'z_score': z_score
        }
    
    def regenerate_forecast(self):
        """Regenerate 30-day forecast with updated data"""
        try:
            self.log("Regenerating forecast...")
            
            # Import and run the forecasting function
            import sys
            sys.path.insert(0, '.')
            
            # Run quick_forecast if available
            try:
                from quick_forecast import quick_forecast
                forecast_df = quick_forecast()
                self.log(f"[OK] Forecast regenerated: {len(forecast_df)} days")
                return forecast_df
            except ImportError:
                self.log("[!] quick_forecast.py not found. Skipping forecast regeneration.")
                self.log("    To enable this, ensure quick_forecast.py exists in project root.")
                return None
            
        except Exception as e:
            self.log(f"[X] Error regenerating forecast: {e}")
            return None
    
    def send_alert(self, anomaly_info):
        """Send alert notification (email/SMS/dashboard)"""
        self.log(f"[ALERT] Sending alert notification...")
        
        # Save alert to file
        alert_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'ANOMALY',
            'pm25_value': anomaly_info['value'],
            'mean': anomaly_info['mean'],
            'std': anomaly_info['std'],
            'z_score': anomaly_info['z_score']
        }
        
        # Save to alerts file
        alerts_file = 'logs/alerts.json'
        
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        else:
            alerts = []
        
        alerts.append(alert_data)
        
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=2)
        
        self.log(f"[OK] Alert saved to {alerts_file}")
    
    def run_daily_update(self):
        """Main daily update routine"""
        self.log("="*60)
        self.log("STARTING DAILY UPDATE PROCESS")
        self.log("="*60)
        
        try:
            # Step 1: Fetch latest data
            self.log("\n[1/5] Fetching latest data...")
            new_row = self.fetch_latest_data()
            
            if not new_row:
                self.log("[X] Failed to fetch data. Aborting update.")
                return False
            
            # Step 2: Append to dataset
            self.log("\n[2/5] Updating dataset...")
            df = self.append_new_data(new_row)
            
            if df is None:
                self.log("[X] Failed to update dataset. Aborting.")
                return False
            
            # Step 3: Check for anomalies
            self.log("\n[3/5] Checking for anomalies...")
            is_anomaly, anomaly_info = self.detect_anomaly(new_row['PM2_5'], df)
            
            # Step 4: Regenerate forecast
            self.log("\n[4/5] Regenerating forecast...")
            forecast_df = self.regenerate_forecast()
            
            if forecast_df is None:
                self.log("[!] Forecast regeneration skipped or failed.")
            
            # Step 5: Send alerts if anomaly
            if is_anomaly:
                self.log("\n[5/5] Sending anomaly alert...")
                self.send_alert(anomaly_info)
            else:
                self.log("\n[5/5] No anomaly detected. No alerts sent.")
            
            self.log("\n" + "="*60)
            self.log("[SUCCESS] DAILY UPDATE COMPLETED")
            self.log("="*60)
            
            return True
            
        except Exception as e:
            self.log(f"\n[CRITICAL ERROR] {e}")
            self.log("="*60)
            return False


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════╗
    ║   PM2.5 AUTO-UPDATE SYSTEM v1.0           ║
    ║   Automatic Daily Forecast Regeneration   ║
    ╚════════════════════════════════════════════╝
    """)
    
    # Create update system
    updater = AutoUpdateSystem()
    
    # Run update
    success = updater.run_daily_update()
    
    if success:
        print("\n[OK] System updated successfully!")
        print("[INFO] New forecast available in: data/outputs/vellore_30day_forecast.csv")
        print("[INFO] Logs saved in: logs/update_log.txt")
    else:
        print("\n[X] Update failed. Check logs for details.")