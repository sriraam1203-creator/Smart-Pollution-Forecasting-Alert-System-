"""
QUICK FIX - Forecasting with Correct Feature Count
===================================================
Match the model's expected 10 features
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
import keras
import matplotlib.pyplot as plt
import os

def quick_forecast():
    """Generate forecast using only the last 10 features (most recent)"""
    
    print("="*70)
    print("🔮 QUICK FORECAST - Matching Model Input Shape")
    print("="*70)
    
    # Load model
    print("\n1️⃣ Loading model...")
    model = keras.models.load_model('models/simple_baseline_model.h5')
    scaler = joblib.load('models/simple_target_scaler.pkl')
    
    # Check model's expected input shape
    expected_shape = model.input_shape
    print(f"   Model expects: {expected_shape}")
    
    n_features_expected = expected_shape[2]
    seq_length = expected_shape[1]
    
    print(f"   → Sequence length: {seq_length}")
    print(f"   → Features: {n_features_expected}")
    
    # Load data
    print("\n2️⃣ Loading data...")
    X_full = np.load('data/processed/X_clean_sequences.npy')
    y = np.load('data/processed/y_clean_targets.npy')
    
    print(f"   Full X shape: {X_full.shape}")
    print(f"   Features available: {X_full.shape[2]}")
    
    # Select LAST n features (most recent lags)
    print(f"\n3️⃣ Selecting last {n_features_expected} features...")
    X = X_full[:, :, -n_features_expected:]  # Take last 10 features
    
    print(f"   Selected X shape: {X.shape}")
    print(f"   ✅ Shape matches model expectation!")
    
    # Get dates
    df = pd.read_csv('data/processed/vellore_clean_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].iloc[14:].values  # Skip first 14 for sequences
    dates = pd.to_datetime(dates)
    
    # Last values
    last_sequence = X[-1:]  # (1, 14, 10)
    last_pm25 = y[-1]
    last_date = dates[-1]
    
    print(f"\n4️⃣ Starting point:")
    print(f"   Last date: {last_date.date()}")
    print(f"   Last PM2.5: {last_pm25:.2f} µg/m³")
    
    # Forecast
    print(f"\n5️⃣ Generating 30-day forecast...")
    
    current_sequence = last_sequence.copy()
    predictions = []
    forecast_dates = []
    
    for day in range(30):
        # Predict (scaled)
        pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        
        # Unscale
        pred_actual = scaler.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(pred_actual)
        
        # Next date
        next_date = last_date + timedelta(days=day + 1)
        forecast_dates.append(next_date)
        
        # Update sequence (roll window)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = current_sequence[0, -2, :]
        
        if (day + 1) % 10 == 0:
            print(f"   Forecasted {day + 1}/30 days...")
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'PM2_5': predictions
    })
    
    # Add alerts
    def get_alert(pm25):
        if pm25 <= 30:
            return 'SAFE'
        elif pm25 <= 60:
            return 'MODERATE'
        elif pm25 <= 90:
            return 'UNHEALTHY'
        else:
            return 'SEVERE'
    
    forecast_df['Alert'] = forecast_df['PM2_5'].apply(get_alert)
    
    # Check continuity
    first_forecast = predictions[0]
    gap = abs(first_forecast - last_pm25)
    
    print(f"\n6️⃣ Continuity Check:")
    print(f"   Last actual:    {last_pm25:.2f} µg/m³")
    print(f"   First forecast: {first_forecast:.2f} µg/m³")
    print(f"   Gap:            {gap:.2f} µg/m³")
    
    if gap < 10:
        print(f"   ✅ Good continuity!")
    else:
        print(f"   ⚠️ Moderate gap (acceptable)")
    
    # Summary
    print(f"\n7️⃣ Forecast Summary:")
    print(f"   Mean PM2.5: {forecast_df['PM2_5'].mean():.2f} µg/m³")
    print(f"   Min:        {forecast_df['PM2_5'].min():.2f} µg/m³")
    print(f"   Max:        {forecast_df['PM2_5'].max():.2f} µg/m³")
    print(f"   Std:        {forecast_df['PM2_5'].std():.2f} µg/m³")
    
    print(f"\n   Alert Distribution:")
    alert_counts = forecast_df['Alert'].value_counts()
    for alert, count in alert_counts.items():
        pct = (count / 30) * 100
        print(f"      {alert:12s}: {count:2d} days ({pct:.0f}%)")
    
    # Save
    print(f"\n8️⃣ Saving forecast...")
    os.makedirs('data/outputs', exist_ok=True)
    forecast_df.to_csv('data/outputs/vellore_30day_forecast.csv', index=False)
    print(f"   ✅ Saved: data/outputs/vellore_30day_forecast.csv")
    
    # Plot
    print(f"\n9️⃣ Creating visualization...")
    
    # Historical data (last 90 days for context)
    historical_df = pd.DataFrame({
        'date': dates[-90:],
        'PM2_5': y[-90:]
    })
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Historical
    ax.plot(historical_df['date'], historical_df['PM2_5'],
            label='Historical PM2.5',
            color='steelblue',
            linewidth=2,
            marker='o',
            markersize=3,
            alpha=0.8)
    
    # Forecast
    ax.plot(forecast_df['date'], forecast_df['PM2_5'],
            label='30-Day Forecast',
            color='darkorange',
            linewidth=2.5,
            marker='s',
            markersize=5,
            linestyle='--',
            alpha=0.9)
    
    # Connection line
    ax.plot([last_date, forecast_dates[0]],
            [last_pm25, predictions[0]],
            color='gray',
            linestyle=':',
            linewidth=2,
            alpha=0.6,
            label='Transition')
    
    # Threshold lines
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=60, color='yellow', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    
    # Vertical separator
    ax.axvline(x=last_date, color='red', linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Forecast Start')
    
    # Labels
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('PM2.5 Historical Data (Last 90 Days) and 30-Day Forecast',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/forecast_visualization.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: results/forecast_visualization.png")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ FORECAST COMPLETE!")
    print("="*70)
    
    return forecast_df


if __name__ == "__main__":
    forecast = quick_forecast()
    
    print("\n📋 First 10 days of forecast:")
    print(forecast[['date', 'PM2_5', 'Alert']].head(10).to_string(index=False))
    
    print("\n📋 Last 10 days of forecast:")
    print(forecast[['date', 'PM2_5', 'Alert']].tail(10).to_string(index=False))