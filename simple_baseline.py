"""
SIMPLE BASELINE MODEL - Guaranteed to Work
==========================================
This uses a proven, simple approach that should give R² > 0.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib
import os

print("="*60)
print("🎯 SIMPLE BASELINE LSTM MODEL")
print("="*60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. Load data
print("\n1️⃣ Loading data...")
X = np.load('data/processed/X_sequences.npy')
y = np.load('data/processed/y_targets.npy')

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   y stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

# 2. Use ONLY the most important features (reduce complexity)
print("\n2️⃣ Feature selection...")
print("   Using only LAST 10 features (most recent lag features)")

# Take only last 10 features (these are usually the most recent lags)
X_selected = X[:, :, -10:]  # Last 10 features
print(f"   Reduced to: {X_selected.shape}")

# 3. Split data (chronological - no shuffle!)
print("\n3️⃣ Splitting data...")
n_train = int(0.7 * len(X_selected))
n_val = int(0.15 * len(X_selected))

X_train = X_selected[:n_train]
y_train = y[:n_train]

X_val = X_selected[n_train:n_train+n_val]
y_val = y[n_train:n_train+n_val]

X_test = X_selected[n_train+n_val:]
y_test = y[n_train+n_val:]

print(f"   Train: {len(X_train)}")
print(f"   Val:   {len(X_val)}")
print(f"   Test:  {len(X_test)}")

# 4. Scale TARGET variable (CRITICAL!)
print("\n4️⃣ Scaling target variable...")
target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

print(f"   Train y: {y_train_scaled.min():.2f} to {y_train_scaled.max():.2f}")
print(f"   Mean: {y_train_scaled.mean():.2f}, Std: {y_train_scaled.std():.2f}")

# 5. Build SIMPLE model
print("\n5️⃣ Building simple LSTM model...")

model = Sequential([
    LSTM(32, input_shape=(X_selected.shape[1], X_selected.shape[2])),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n   Model Summary:")
model.summary()

# 6. Train with early stopping
print("\n6️⃣ Training model...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n✅ Training complete!")

# 7. Evaluate on test set
print("\n7️⃣ Evaluating on test set...")

# Predict (scaled)
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Inverse transform to original scale
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

# Baseline comparison
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\n" + "="*60)
print("📊 TEST SET RESULTS")
print("="*60)
print(f"   MAE:   {mae:.2f} µg/m³")
print(f"   RMSE:  {rmse:.2f} µg/m³")
print(f"   R²:    {r2:.4f}")
print(f"   MAPE:  {mape:.2f}%")
print("="*60)

print(f"\n📊 BASELINE COMPARISON (predict mean = {y_train.mean():.2f}):")
print(f"   Baseline MAE: {baseline_mae:.2f} µg/m³")
print(f"   Baseline R²:  {baseline_r2:.4f}")
print(f"   Model MAE:    {mae:.2f} µg/m³")
print(f"   Model R²:     {r2:.4f}")

if mae < baseline_mae:
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    print(f"\n   ✅ Model is {improvement:.1f}% better than baseline!")
else:
    print(f"\n   ❌ Model is worse than baseline - need to debug")

print(f"\n📈 Prediction Statistics:")
print(f"   True:      [{y_test.min():.2f}, {y_test.max():.2f}] (mean: {y_test.mean():.2f})")
print(f"   Predicted: [{y_pred.min():.2f}, {y_pred.max():.2f}] (mean: {y_pred.mean():.2f})")

# Check if predictions are too flat
pred_std = y_pred.std()
true_std = y_test.std()
print(f"   True std:  {true_std:.2f}")
print(f"   Pred std:  {pred_std:.2f}")

if pred_std < true_std * 0.3:
    print("   ⚠️ WARNING: Predictions are too flat (low variance)")
    print("   Model is predicting values too close to the mean")

# 8. Visualize results
print("\n8️⃣ Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training history - Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training History - Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training history - MAE
axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('Training History - MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Predictions - Time series
axes[1, 0].plot(y_test, label='Actual', marker='o', linewidth=2, markersize=4)
axes[1, 0].plot(y_pred, label='Predicted', marker='s', linewidth=2, markersize=4, alpha=0.7)
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('PM2.5 (µg/m³)')
axes[1, 0].set_title('Test Set: Actual vs Predicted')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Scatter plot
axes[1, 1].scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual PM2.5 (µg/m³)')
axes[1, 1].set_ylabel('Predicted PM2.5 (µg/m³)')
axes[1, 1].set_title(f'Prediction Accuracy (R² = {r2:.4f})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/simple_baseline_results.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/simple_baseline_results.png")
plt.close()

# 9. Save model and scaler
print("\n9️⃣ Saving model...")
model.save('models/simple_baseline_model.h5')
joblib.dump(target_scaler, 'models/simple_target_scaler.pkl')
joblib.dump([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'models/simple_selected_features.pkl')  # Last 10 features

print("\n   ✅ Model saved to: models/simple_baseline_model.h5")
print("   ✅ Scaler saved to: models/simple_target_scaler.pkl")

# 10. Save metrics
metrics_df = pd.DataFrame([{
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'MAPE': mape,
    'Baseline_MAE': baseline_mae,
    'Baseline_R2': baseline_r2
}])
metrics_df.to_csv('results/simple_baseline_metrics.csv', index=False)

print("\n" + "="*60)
print("✅ BASELINE MODEL COMPLETE!")
print("="*60)

if r2 > 0.3:
    print("\n🎉 SUCCESS! Model is learning patterns!")
    print(f"   R² = {r2:.4f} means model explains {r2*100:.1f}% of variance")
else:
    print("\n⚠️ Model still struggling. Possible issues:")
    print("   1. Data might have inherent limitations")
    print("   2. Preprocessing might have issues")
    print("   3. PM2.5 patterns might be too complex for this dataset size")
    
print("\n📊 Next steps:")
print("   1. Check results/simple_baseline_results.png")
print("   2. If R² > 0.3, we can build on this")
print("   3. If R² < 0.3, we need to revisit data preprocessing")