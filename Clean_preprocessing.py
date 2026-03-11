"""
CLEAN PREPROCESSING - NO TARGET LEAKAGE
========================================
This version ensures no future information leaks into features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def create_clean_features(df):
    """Create features WITHOUT target leakage"""
    
    print("🔧 Creating features (leak-free)...")
    df = df.copy()
    
    # ========================================
    # 1. TEMPORAL FEATURES (Safe - no leakage)
    # ========================================
    print("   1️⃣ Temporal features...")
    
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical encoding
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Season
    df['season'] = df['month'].apply(lambda x: 
        0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3)
    
    # ========================================
    # 2. LAG FEATURES (Safe - using PAST only)
    # ========================================
    print("   2️⃣ Lag features (past values only)...")
    
    # CRITICAL: Use shift() to get PAST values only
    for lag in [1, 2, 3, 7, 14]:
        df[f'PM2_5_lag_{lag}'] = df['PM2_5'].shift(lag)  # Past value
    
    if 'AOD' in df.columns:
        for lag in [1, 3, 7]:
            df[f'AOD_lag_{lag}'] = df['AOD'].shift(lag)
    
    # Weather lags
    for var in ['temperature', 'humidity', 'wind_speed']:
        if var in df.columns:
            for lag in [1, 3]:
                df[f'{var}_lag_{lag}'] = df[var].shift(lag)
    
    # ========================================
    # 3. ROLLING FEATURES (Safe - PAST window only)
    # ========================================
    print("   3️⃣ Rolling statistics (past windows only)...")
    
    # CRITICAL: These must NOT include current value
    # Use shift(1) before rolling to exclude current day
    pm25_shifted = df['PM2_5'].shift(1)
    
    for window in [3, 7, 14]:
        df[f'PM2_5_roll_mean_{window}'] = pm25_shifted.rolling(window, min_periods=1).mean()
        df[f'PM2_5_roll_std_{window}'] = pm25_shifted.rolling(window, min_periods=1).std()
        df[f'PM2_5_roll_min_{window}'] = pm25_shifted.rolling(window, min_periods=1).min()
        df[f'PM2_5_roll_max_{window}'] = pm25_shifted.rolling(window, min_periods=1).max()
    
    # ========================================
    # 4. WEATHER INTERACTIONS (Safe - current weather is OK)
    # ========================================
    print("   4️⃣ Weather interactions...")
    
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interact'] = df['temperature'] * df['humidity']
    
    if 'wind_speed' in df.columns and 'AOD' in df.columns:
        df['wind_aod_interact'] = df['wind_speed'] * df['AOD']
    
    # ========================================
    # 5. TREND FEATURES (Safe - using past only)
    # ========================================
    print("   5️⃣ Trend features...")
    
    df['PM2_5_trend_3d'] = df['PM2_5'] - df['PM2_5'].shift(3)
    df['PM2_5_trend_7d'] = df['PM2_5'] - df['PM2_5'].shift(7)
    
    print(f"   ✅ Created {len(df.columns)} total columns")
    
    return df


def clean_preprocessing_pipeline():
    """Run clean preprocessing WITHOUT target leakage"""
    
    print("\n" + "="*60)
    print("🚀 CLEAN PREPROCESSING (NO TARGET LEAKAGE)")
    print("="*60)
    
    # Load data
    print("\n1️⃣ Loading data...")
    df_sat = pd.read_csv('data/raw/satellite/vellore_pm_from_aod.csv')
    df_sat['date'] = pd.to_datetime(df_sat['date'])
    
    df_weather = pd.read_csv('data/raw/weather/vellore_weather_raw.csv')
    df_weather['date'] = pd.to_datetime(df_weather['datetime']).dt.date
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    # Aggregate weather to daily
    df_weather_daily = df_weather.groupby('date').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    print(f"   Satellite: {len(df_sat)} rows")
    print(f"   Weather: {len(df_weather_daily)} rows")
    
    # Merge
    print("\n2️⃣ Merging datasets...")
    merged = pd.merge(df_sat, df_weather_daily, on='date', how='inner')
    merged = merged.fillna(method='ffill').fillna(method='bfill')
    print(f"   Merged: {len(merged)} rows")
    
    # Create clean features
    print("\n3️⃣ Feature engineering (leak-free)...")
    merged = create_clean_features(merged)
    
    # Drop NaN rows from lags
    print("\n4️⃣ Cleaning data...")
    print(f"   Before: {len(merged)} rows")
    merged = merged.dropna().reset_index(drop=True)
    print(f"   After: {len(merged)} rows")
    
    # Select features (EXCLUDE PM2.5 from features!)
    print("\n5️⃣ Selecting features...")
    exclude_cols = ['date', 'PM2_5', 'year', 'datetime', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    feature_cols = [col for col in merged.columns if col not in exclude_cols]
    
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Features: {feature_cols[:10]}... (showing first 10)")
    
    # Normalize features only (NOT target!)
    print("\n6️⃣ Normalizing features...")
    scaler = MinMaxScaler()
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols])
    
    # Create sequences
    print("\n7️⃣ Creating sequences...")
    seq_length = 14
    
    X, y = [], []
    data = merged[feature_cols].values
    target = merged['PM2_5'].values  # Keep target in original scale!
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}] µg/m³")
    
    # Verification check
    print("\n8️⃣ Verification check...")
    print(f"   Features scaled: [{merged[feature_cols].min().min():.4f}, {merged[feature_cols].max().max():.4f}]")
    print(f"   Target NOT scaled: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   ✅ No target leakage: PM2_5 NOT in feature list")
    
    # Save
    print("\n9️⃣ Saving...")
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    merged.to_csv('data/processed/vellore_clean_dataset.csv', index=False)
    np.save('data/processed/X_clean_sequences.npy', X)
    np.save('data/processed/y_clean_targets.npy', y)
    joblib.dump(scaler, 'models/clean_scaler.pkl')
    joblib.dump(feature_cols, 'models/clean_feature_list.pkl')
    
    print("\n" + "="*60)
    print("✅ CLEAN PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Final rows: {len(merged)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Sequences: {len(X)}")
    print(f"\n🔒 Leak Protection:")
    print(f"   ✅ PM2.5 NOT in features")
    print(f"   ✅ Lags use shift() (past only)")
    print(f"   ✅ Rolling windows exclude current day")
    print(f"   ✅ Target NOT normalized (stays in µg/m³)")
    
    print(f"\n🚀 Next: Run optimized model with clean data!")
    
    return X, y, merged, feature_cols


if __name__ == "__main__":
    X, y, df, features = clean_preprocessing_pipeline()