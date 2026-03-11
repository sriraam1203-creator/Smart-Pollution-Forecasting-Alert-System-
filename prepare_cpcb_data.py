"""
Prepare CPCB Data for Model Training
====================================
Converts CPCB raw data to model-ready format
"""

import pandas as pd
import numpy as np
import os

def prepare_cpcb_data(input_file):
    """
    Prepare CPCB pollution data
    
    Args:
        input_file: Path to downloaded CPCB CSV file
    """
    
    print("="*60)
    print("🔧 PREPARING CPCB DATA")
    print("="*60)
    
    # Load data
    print(f"\n1️⃣ Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"   ✅ Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # Check timestamp column
    print("\n2️⃣ Processing dates...")
    timestamp_col = None
    for col in ['Timestamp', 'timestamp', 'Date', 'date', 'From Date']:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("   ❌ No timestamp column found!")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=['date'])
    
    print(f"   ✅ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Extract required columns
    print("\n3️⃣ Extracting pollutant data...")
    
    # Map column names (CPCB format to our format)
    column_mapping = {
        'PM2.5 (µg/m³)': 'PM2_5',
        'PM2.5': 'PM2_5',
        'PM10 (µg/m³)': 'PM10',
        'PM10': 'PM10',
        'NO2 (µg/m³)': 'NO2',
        'NO2': 'NO2',
        'SO2 (µg/m³)': 'SO2',
        'SO2': 'SO2',
        'CO (mg/m³)': 'CO',
        'CO': 'CO',
        'Ozone (µg/m³)': 'O3',
        'Ozone': 'O3',
        'AT (°C)': 'Temperature',
        'AT': 'Temperature',
        'RH (%)': 'Humidity',
        'RH': 'Humidity',
        'WS (m/s)': 'WindSpeed',
        'WS': 'WindSpeed'
    }
    
    # Rename columns
    df_clean = pd.DataFrame({'date': df['date']})
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df_clean[new_name] = pd.to_numeric(df[old_name], errors='coerce')
            print(f"   ✅ {new_name}")
    
    # Calculate AOD from PM2.5 (approximate relationship)
    if 'PM2_5' in df_clean.columns:
        df_clean['AOD'] = df_clean['PM2_5'] / 150  # Approximate conversion
        print(f"   ✅ AOD (calculated from PM2.5)")
    
    # Aggregate to daily (in case there are hourly readings)
    print("\n4️⃣ Aggregating to daily averages...")
    df_daily = df_clean.groupby('date').agg({
        col: 'mean' for col in df_clean.columns if col != 'date'
    }).reset_index()
    
    print(f"   ✅ {len(df_daily)} unique days")
    
    # Handle missing values
    print("\n5️⃣ Handling missing values...")
    print(f"   Missing values before:")
    missing_before = df_daily.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Forward fill and backward fill
    df_daily = df_daily.fillna(method='ffill').fillna(method='bfill')
    
    missing_after = df_daily.isnull().sum().sum()
    if missing_after > 0:
        print(f"   ⚠️ Still {missing_after} missing values - dropping those rows")
        df_daily = df_daily.dropna()
    else:
        print(f"   ✅ All missing values filled")
    
    # Save processed data
    print("\n6️⃣ Saving processed data...")
    
    # Save satellite data (PM2.5, PM10, AOD, NO2)
    satellite_cols = ['date', 'PM2_5', 'PM10', 'AOD']
    if 'NO2' in df_daily.columns:
        satellite_cols.append('NO2')
    
    os.makedirs('data/raw/satellite', exist_ok=True)
    satellite_file = 'data/raw/satellite/vellore_pm_from_aod.csv'
    df_daily[satellite_cols].to_csv(satellite_file, index=False)
    print(f"   ✅ Satellite data: {satellite_file}")
    
    # If weather data is in CPCB file, save separately
    weather_cols = ['date']
    if 'Temperature' in df_daily.columns:
        weather_cols.append('Temperature')
    if 'Humidity' in df_daily.columns:
        weather_cols.append('Humidity')
    if 'WindSpeed' in df_daily.columns:
        weather_cols.append('WindSpeed')
    
    if len(weather_cols) > 1:
        # Rename to match our format
        weather_df = df_daily[weather_cols].copy()
        weather_df.rename(columns={
            'date': 'datetime',
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'WindSpeed': 'wind_speed'
        }, inplace=True)
        weather_df['weather'] = 'clear'
        
        weather_file = 'data/raw/weather/vellore_weather_cpcb.csv'
        weather_df.to_csv(weather_file, index=False)
        print(f"   ✅ Weather data: {weather_file}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Total days: {len(df_daily)}")
    print(f"   Date range: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")
    print(f"   Columns: {list(df_daily.columns)}")
    
    print("\n📋 Data preview:")
    print(df_daily.head())
    
    print("\n🚀 Next steps:")
    print("   1. Merge with weather data (if needed)")
    print("   2. Run: python enhanced_features.py")
    print("   3. Run: python lstm_model_training.py")
    
    return df_daily


if __name__ == "__main__":
    # Your CPCB file path
    input_file = r"C:\Users\vijai\Downloads\Raw_data_1Day_2024_site_5615_Vasanthapuram_Vellore_TNPCB_1Day.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
    else:
        df = prepare_cpcb_data(input_file)