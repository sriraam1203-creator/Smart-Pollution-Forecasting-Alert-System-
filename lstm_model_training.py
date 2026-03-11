import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import os

class PM25LSTMModel:
    """
    FIXED LSTM Model for PM2.5 Forecasting
    """
    
    def __init__(self, seq_length=14, n_features=75):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.target_scaler = MinMaxScaler()  # Separate scaler for target
        
    def build_model(self, lstm_units=[128, 64], dropout=0.3):
        """Build IMPROVED LSTM architecture"""
        model = Sequential()
        
        # First Bidirectional LSTM layer (better for time series)
        model.add(Bidirectional(LSTM(lstm_units[0], 
                                     return_sequences=True,
                                     kernel_regularizer=keras.regularizers.l2(0.001)),
                               input_shape=(self.seq_length, self.n_features)))
        model.add(Dropout(dropout))
        
        # Second LSTM layer
        model.add(LSTM(lstm_units[1], 
                      return_sequences=False,
                      kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Dropout(dropout))
        
        # Dense layers for better representation
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout/2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile with better optimizer settings
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        print("\n🏗️ IMPROVED Model Architecture:")
        model.summary()
        
        return model
    
    def prepare_data_with_scaling(self, X, y):
        """Normalize target variable - THIS WAS MISSING!"""
        print("\n📊 Scaling target variable...")
        print(f"   Original y range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Scale target to 0-1 range
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        print(f"   Scaled y range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
        
        return X, y_scaled
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=150, batch_size=16, verbose=1):
        """Train the LSTM model with better callbacks"""
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'models/best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        print("\n🚀 Starting training...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=verbose
        )
        
        print("✅ Training complete!")
        
        return self.history
    
    def evaluate(self, X_test, y_test_scaled):
        """Evaluate model performance with CORRECT inverse transform"""
        
        print("\n📊 Evaluating model...")
        
        # Predictions (scaled)
        y_pred_scaled = self.model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform to original scale - THIS IS THE FIX!
        y_test_original = self.target_scaler.inverse_transform(
            y_test_scaled.reshape(-1, 1)
        ).flatten()
        
        y_pred_original = self.target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        
        # Calculate metrics on ORIGINAL scale
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        
        # Additional metrics
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION METRICS (Original Scale)")
        print("="*60)
        print(f"   MAE:   {mae:.2f} µg/m³")
        print(f"   RMSE:  {rmse:.2f} µg/m³")
        print(f"   R²:    {r2:.4f}")
        print(f"   MAPE:  {mape:.2f}%")
        print("="*60)
        
        # Show prediction range
        print(f"\n📈 Prediction Statistics:")
        print(f"   True values:      [{y_test_original.min():.2f}, {y_test_original.max():.2f}] µg/m³")
        print(f"   Predicted values: [{y_pred_original.min():.2f}, {y_pred_original.max():.2f}] µg/m³")
        print(f"   Mean true:        {y_test_original.mean():.2f} µg/m³")
        print(f"   Mean predicted:   {y_pred_original.mean():.2f} µg/m³")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'y_true': y_test_original,
            'y_pred': y_pred_original
        }
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (Huber)', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Training & Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: results/training_history.png")
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, n_samples=100):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series comparison
        indices = np.arange(min(n_samples, len(y_true)))
        axes[0].plot(indices, y_true[:n_samples], 
                    label='Actual PM2.5', marker='o', linewidth=2, markersize=4)
        axes[0].plot(indices, y_pred[:n_samples], 
                    label='Predicted PM2.5', marker='s', linewidth=2, markersize=4, alpha=0.7)
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=12)
        axes[0].set_title(f'Actual vs Predicted PM2.5 (First {n_samples} samples)', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', lw=3, label='Perfect Prediction', alpha=0.8)
        
        axes[1].set_xlabel('Actual PM2.5 (µg/m³)', fontsize=12)
        axes[1].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=12)
        axes[1].set_title('Prediction Accuracy Scatter Plot', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[1].transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/predictions_plot.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: results/predictions_plot.png")
        plt.close()
    
    def save_model(self, filepath='models/final_lstm_model.keras'):
        """Save trained model and scaler"""
        self.model.save(filepath)
        joblib.dump(self.target_scaler, 'models/target_scaler.pkl')
        print(f"\n💾 Model saved to {filepath}")
        print(f"💾 Target scaler saved to models/target_scaler.pkl")


# Complete training pipeline
def train_pipeline():
    """Complete FIXED training workflow"""
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 PM2.5 FORECASTING - FIXED LSTM TRAINING PIPELINE")
    print("="*60)
    
    # Load preprocessed data
    print("\n1️⃣ Loading preprocessed data...")
    X = np.load('data/processed/X_sequences.npy')
    y = np.load('data/processed/y_targets.npy')
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}] µg/m³")
    
    # Check for issues
    if np.isnan(X).any() or np.isnan(y).any():
        print("   ⚠️ WARNING: NaN values detected in data!")
        return None
    
    # Split data
    print("\n2️⃣ Splitting data (70% train, 15% val, 15% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, shuffle=False  # 0.176 * 0.85 ≈ 0.15 of total
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Build model
    print("\n3️⃣ Building IMPROVED LSTM model...")
    seq_length = X.shape[1]
    n_features = X.shape[2]
    
    lstm_model = PM25LSTMModel(seq_length=seq_length, n_features=n_features)
    lstm_model.build_model(lstm_units=[128, 64], dropout=0.3)
    
    # Scale target variable - THIS IS THE CRITICAL FIX!
    print("\n4️⃣ Scaling target variable...")
    X_train, y_train_scaled = lstm_model.prepare_data_with_scaling(X_train, y_train)
    
    # Use the SAME scaler for val and test
    y_val_scaled = lstm_model.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = lstm_model.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Train
    print("\n5️⃣ Training model...")
    lstm_model.train(
        X_train, y_train_scaled,
        X_val, y_val_scaled,
        epochs=150,
        batch_size=16
    )
    
    # Evaluate
    print("\n6️⃣ Evaluating model...")
    results = lstm_model.evaluate(X_test, y_test_scaled)
    
    # Plot results
    print("\n7️⃣ Generating plots...")
    lstm_model.plot_training_history()
    lstm_model.plot_predictions(results['y_true'], results['y_pred'])
    
    # Save model
    print("\n8️⃣ Saving model...")
    lstm_model.save_model('models/final_lstm_model.h5')
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'MAE': results['mae'],
        'RMSE': results['rmse'],
        'R2': results['r2'],
        'MAPE': results['mape']
    }])
    metrics_df.to_csv('results/evaluation_metrics.csv', index=False)
    print("   ✅ Saved: results/evaluation_metrics.csv")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   MAE: {results['mae']:.2f} µg/m³")
    print(f"   RMSE: {results['rmse']:.2f} µg/m³")
    print(f"   R²: {results['r2']:.4f}")
    
    return lstm_model, results


if __name__ == "__main__":
    model, results = train_pipeline()