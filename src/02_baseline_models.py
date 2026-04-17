"""
================================================================================
MISSION 2: BASELINE MODELING
================================================================================
Research Project: Federated Learning for CO2 Prediction in Smart School Environments
Target: Q1 Journal Publication

This script establishes baseline results before federated learning comparison:
1. Centralized LSTM (all schools combined)
2. Local-only LSTM models (one per school)

All models use identical architecture and training settings for fair comparison.

Author: Research Team
Date: March 2024
================================================================================
"""

import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import matplotlib.pyplot as plt

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
BASE_PATH = r"C:\Users\info\Documents\my-project\Federated Learning"
DATA_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis")
OUTPUT_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis", "mission2_results")

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Plot settings
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 80)
print("MISSION 2: BASELINE MODELING")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Random seed: {RANDOM_SEED}")

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    # Model Architecture (Fixed for all experiments)
    'lstm_units': 64,
    'dropout_rate': 0.2,

    # Training Settings (Fixed for all experiments)
    'learning_rate': 0.001,
    'batch_size': 32,
    'max_epochs': 50,
    'patience': 8,

    # Data
    'window_size': 12,
    'n_features': 5,

    # Reproducibility
    'random_seed': RANDOM_SEED
}

print("\n" + "-" * 60)
print("CONFIGURATION (Fixed for All Models)")
print("-" * 60)
print(f"LSTM Units: {CONFIG['lstm_units']}")
print(f"Dropout Rate: {CONFIG['dropout_rate']}")
print(f"Learning Rate: {CONFIG['learning_rate']}")
print(f"Batch Size: {CONFIG['batch_size']}")
print(f"Max Epochs: {CONFIG['max_epochs']}")
print(f"Early Stopping Patience: {CONFIG['patience']}")

# ================================================================================
# LOAD DATA
# ================================================================================

print("\n" + "=" * 80)
print("LOADING PREPARED DATA FROM MISSION 1")
print("=" * 80)

def load_school_data(school):
    """Load train/val/test data for a school."""
    data = {}
    for split in ['train', 'val', 'test']:
        X = np.load(os.path.join(DATA_PATH, f"final_X_{school}_{split}.npy"))
        y = np.load(os.path.join(DATA_PATH, f"final_y_{school}_{split}.npy"))
        data[split] = {'X': X, 'y': y}
    return data

# Load data for each school
school_data = {}
for school in ['A', 'B', 'C']:
    school_data[school] = load_school_data(school)
    print(f"\nSchool {school}:")
    for split in ['train', 'val', 'test']:
        print(f"  {split}: X={school_data[school][split]['X'].shape}, y={school_data[school][split]['y'].shape}")

# Load scaler parameters for inverse transform
with open(os.path.join(DATA_PATH, "final_scaler_params.json"), 'r') as f:
    scaler_params = json.load(f)

CO2_MIN = scaler_params['co2_inverse_transform']['min_ppm']
CO2_MAX = scaler_params['co2_inverse_transform']['max_ppm']

print(f"\nCO2 Inverse Transform: y_ppm = y_norm * ({CO2_MAX} - {CO2_MIN}) + {CO2_MIN}")

def inverse_transform_co2(y_normalized):
    """Convert normalized CO2 values back to ppm."""
    return y_normalized * (CO2_MAX - CO2_MIN) + CO2_MIN

# ================================================================================
# PREPARE CENTRALIZED DATA
# ================================================================================

print("\n" + "=" * 80)
print("PREPARING CENTRALIZED DATASET")
print("=" * 80)

# Combine all schools' data
centralized_data = {}
for split in ['train', 'val', 'test']:
    X_combined = np.concatenate([school_data[s][split]['X'] for s in ['A', 'B', 'C']], axis=0)
    y_combined = np.concatenate([school_data[s][split]['y'] for s in ['A', 'B', 'C']], axis=0)
    centralized_data[split] = {'X': X_combined, 'y': y_combined}
    print(f"Centralized {split}: X={X_combined.shape}, y={y_combined.shape}")

# ================================================================================
# MODEL DEFINITION
# ================================================================================

def create_lstm_model(input_shape, config):
    """
    Create LSTM model with fixed architecture.

    Architecture:
    - Input layer
    - LSTM layer (64 units)
    - Dropout (0.2)
    - Dense output layer (1 unit)

    This architecture is intentionally simple and robust,
    suitable for baseline comparison in federated learning research.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config['lstm_units'], return_sequences=False),
        Dropout(config['dropout_rate']),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='mse',
        metrics=['mae']
    )

    return model

# Display model architecture
print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)

input_shape = (CONFIG['window_size'], CONFIG['n_features'])
sample_model = create_lstm_model(input_shape, CONFIG)
sample_model.summary()

# ================================================================================
# TRAINING FUNCTION
# ================================================================================

def train_model(model, X_train, y_train, X_val, y_val, config, model_name, save_path):
    """
    Train model with early stopping and save best weights.

    Returns:
    --------
    history : keras History object
    best_epoch : int
    """
    print(f"\nTraining {model_name}...")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config['patience'],
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, f"{model_name}_best.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['max_epochs'],
        batch_size=config['batch_size'],
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Find best epoch
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val_loss: {best_val_loss:.6f}")

    return history, best_epoch

# ================================================================================
# EVALUATION FUNCTION
# ================================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and return metrics in both normalized and ppm scale.

    Returns:
    --------
    metrics : dict with RMSE, MAE, R2 in ppm
    y_pred_ppm : predictions in ppm
    y_true_ppm : ground truth in ppm
    """
    # Predict (normalized)
    y_pred_norm = model.predict(X_test, verbose=0).flatten()

    # Inverse transform to ppm
    y_pred_ppm = inverse_transform_co2(y_pred_norm)
    y_true_ppm = inverse_transform_co2(y_test)

    # Calculate metrics in ppm
    rmse = np.sqrt(mean_squared_error(y_true_ppm, y_pred_ppm))
    mae = mean_absolute_error(y_true_ppm, y_pred_ppm)
    r2 = r2_score(y_true_ppm, y_pred_ppm)

    metrics = {
        'RMSE_ppm': rmse,
        'MAE_ppm': mae,
        'R2': r2
    }

    print(f"\n{model_name} Test Results:")
    print(f"  RMSE: {rmse:.2f} ppm")
    print(f"  MAE:  {mae:.2f} ppm")
    print(f"  R2:   {r2:.4f}")

    return metrics, y_pred_ppm, y_true_ppm

# ================================================================================
# EXPERIMENT 1: CENTRALIZED MODEL
# ================================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: CENTRALIZED LSTM MODEL")
print("=" * 80)

# Reset seeds before each experiment
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create and train centralized model
centralized_model = create_lstm_model(input_shape, CONFIG)

centralized_history, centralized_best_epoch = train_model(
    centralized_model,
    centralized_data['train']['X'],
    centralized_data['train']['y'],
    centralized_data['val']['X'],
    centralized_data['val']['y'],
    CONFIG,
    "Centralized_LSTM",
    OUTPUT_PATH
)

# Evaluate on combined test set
centralized_metrics, centralized_pred, centralized_true = evaluate_model(
    centralized_model,
    centralized_data['test']['X'],
    centralized_data['test']['y'],
    "Centralized Model"
)

# Also evaluate centralized model on each school's test set separately
print("\nCentralized Model - Per-School Test Performance:")
centralized_per_school = {}
for school in ['A', 'B', 'C']:
    metrics, _, _ = evaluate_model(
        centralized_model,
        school_data[school]['test']['X'],
        school_data[school]['test']['y'],
        f"Centralized on School {school}"
    )
    centralized_per_school[school] = metrics

# ================================================================================
# EXPERIMENT 2: LOCAL-ONLY MODELS
# ================================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: LOCAL-ONLY LSTM MODELS")
print("=" * 80)

local_models = {}
local_histories = {}
local_metrics = {}
local_predictions = {}

for school in ['A', 'B', 'C']:
    print(f"\n{'-' * 40}")
    print(f"Training Local Model for School {school}")
    print(f"{'-' * 40}")

    # Reset seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Create model
    model = create_lstm_model(input_shape, CONFIG)

    # Train
    history, best_epoch = train_model(
        model,
        school_data[school]['train']['X'],
        school_data[school]['train']['y'],
        school_data[school]['val']['X'],
        school_data[school]['val']['y'],
        CONFIG,
        f"Local_{school}_LSTM",
        OUTPUT_PATH
    )

    # Evaluate
    metrics, y_pred, y_true = evaluate_model(
        model,
        school_data[school]['test']['X'],
        school_data[school]['test']['y'],
        f"Local Model {school}"
    )

    local_models[school] = model
    local_histories[school] = history
    local_metrics[school] = metrics
    local_predictions[school] = {'pred': y_pred, 'true': y_true}

# ================================================================================
# RESULTS SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Create comparison table
print("\n" + "-" * 70)
print("COMPARISON TABLE: ALL BASELINE MODELS")
print("-" * 70)
print(f"{'Model':<25} {'RMSE (ppm)':>12} {'MAE (ppm)':>12} {'R2':>10}")
print("-" * 70)

# Centralized results
print(f"{'Centralized (All)':<25} {centralized_metrics['RMSE_ppm']:>12.2f} {centralized_metrics['MAE_ppm']:>12.2f} {centralized_metrics['R2']:>10.4f}")

# Local results
for school in ['A', 'B', 'C']:
    m = local_metrics[school]
    print(f"{'Local ' + school:<25} {m['RMSE_ppm']:>12.2f} {m['MAE_ppm']:>12.2f} {m['R2']:>10.4f}")

print("-" * 70)

# Centralized per-school performance
print("\n" + "-" * 70)
print("CENTRALIZED MODEL - PER-SCHOOL PERFORMANCE")
print("-" * 70)
print(f"{'Test Set':<25} {'RMSE (ppm)':>12} {'MAE (ppm)':>12} {'R2':>10}")
print("-" * 70)
for school in ['A', 'B', 'C']:
    m = centralized_per_school[school]
    print(f"{'School ' + school + ' Test':<25} {m['RMSE_ppm']:>12.2f} {m['MAE_ppm']:>12.2f} {m['R2']:>10.4f}")
print("-" * 70)

# ================================================================================
# VISUALIZATIONS
# ================================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Training and Validation Loss Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Centralized
ax = axes[0, 0]
ax.plot(centralized_history.history['loss'], label='Train Loss', linewidth=2)
ax.plot(centralized_history.history['val_loss'], label='Val Loss', linewidth=2)
ax.axvline(centralized_best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({centralized_best_epoch})')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Centralized Model')
ax.legend()
ax.grid(True, alpha=0.3)

# Local models
for idx, school in enumerate(['A', 'B', 'C']):
    ax = axes[(idx + 1) // 2, (idx + 1) % 2]
    history = local_histories[school]
    best_epoch = np.argmin(history.history['val_loss']) + 1

    ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax.axvline(best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Local Model - School {school}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'loss_curves.png'), dpi=300)
plt.close()
print("   Saved: loss_curves.png")

# 2. Prediction vs Ground Truth (Scatter plots)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Centralized
ax = axes[0, 0]
ax.scatter(centralized_true, centralized_pred, alpha=0.3, s=10)
ax.plot([CO2_MIN, CO2_MAX], [CO2_MIN, CO2_MAX], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Ground Truth (ppm)')
ax.set_ylabel('Prediction (ppm)')
ax.set_title(f'Centralized Model\nRMSE={centralized_metrics["RMSE_ppm"]:.1f}, R2={centralized_metrics["R2"]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Local models
for idx, school in enumerate(['A', 'B', 'C']):
    ax = axes[(idx + 1) // 2, (idx + 1) % 2]
    pred = local_predictions[school]['pred']
    true = local_predictions[school]['true']
    m = local_metrics[school]

    ax.scatter(true, pred, alpha=0.3, s=10)
    ax.plot([CO2_MIN, CO2_MAX], [CO2_MIN, CO2_MAX], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Ground Truth (ppm)')
    ax.set_ylabel('Prediction (ppm)')
    ax.set_title(f'Local Model - School {school}\nRMSE={m["RMSE_ppm"]:.1f}, R2={m["R2"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Prediction vs Ground Truth', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'prediction_scatter.png'), dpi=300)
plt.close()
print("   Saved: prediction_scatter.png")

# 3. Time Series Comparison (Sample of predictions)
fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# Sample range for visualization
sample_range = 500

# Centralized
ax = axes[0]
ax.plot(centralized_true[:sample_range], label='Ground Truth', alpha=0.8, linewidth=1)
ax.plot(centralized_pred[:sample_range], label='Prediction', alpha=0.8, linewidth=1)
ax.set_ylabel('CO2 (ppm)')
ax.set_title('Centralized Model - Time Series Prediction')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Local models
for idx, school in enumerate(['A', 'B', 'C']):
    ax = axes[idx + 1]
    pred = local_predictions[school]['pred']
    true = local_predictions[school]['true']

    ax.plot(true[:sample_range], label='Ground Truth', alpha=0.8, linewidth=1)
    ax.plot(pred[:sample_range], label='Prediction', alpha=0.8, linewidth=1)
    ax.set_ylabel('CO2 (ppm)')
    ax.set_title(f'Local Model School {school} - Time Series Prediction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Sample Index')
plt.suptitle('Time Series: Prediction vs Ground Truth (First 500 Samples)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'time_series_comparison.png'), dpi=300)
plt.close()
print("   Saved: time_series_comparison.png")

# 4. Bar Chart Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = ['Centralized', 'Local A', 'Local B', 'Local C']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

# RMSE
ax = axes[0]
rmse_values = [centralized_metrics['RMSE_ppm']] + [local_metrics[s]['RMSE_ppm'] for s in ['A', 'B', 'C']]
bars = ax.bar(models, rmse_values, color=colors)
ax.set_ylabel('RMSE (ppm)')
ax.set_title('RMSE Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=10)

# MAE
ax = axes[1]
mae_values = [centralized_metrics['MAE_ppm']] + [local_metrics[s]['MAE_ppm'] for s in ['A', 'B', 'C']]
bars = ax.bar(models, mae_values, color=colors)
ax.set_ylabel('MAE (ppm)')
ax.set_title('MAE Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=10)

# R2
ax = axes[2]
r2_values = [centralized_metrics['R2']] + [local_metrics[s]['R2'] for s in ['A', 'B', 'C']]
bars = ax.bar(models, r2_values, color=colors)
ax.set_ylabel('R2 Score')
ax.set_title('R2 Comparison')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)

plt.suptitle('Baseline Model Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'metrics_comparison.png'), dpi=300)
plt.close()
print("   Saved: metrics_comparison.png")

# ================================================================================
# SAVE RESULTS
# ================================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Compile all results
results = {
    'config': CONFIG,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'centralized': {
        'metrics': centralized_metrics,
        'per_school_metrics': centralized_per_school,
        'best_epoch': centralized_best_epoch,
        'train_samples': len(centralized_data['train']['y']),
        'test_samples': len(centralized_data['test']['y'])
    },
    'local': {
        school: {
            'metrics': local_metrics[school],
            'best_epoch': int(np.argmin(local_histories[school].history['val_loss']) + 1),
            'train_samples': len(school_data[school]['train']['y']),
            'test_samples': len(school_data[school]['test']['y'])
        }
        for school in ['A', 'B', 'C']
    }
}

with open(os.path.join(OUTPUT_PATH, 'baseline_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("   Saved: baseline_results.json")

# Save training histories
histories_data = {
    'centralized': {
        'loss': centralized_history.history['loss'],
        'val_loss': centralized_history.history['val_loss']
    },
    'local': {
        school: {
            'loss': local_histories[school].history['loss'],
            'val_loss': local_histories[school].history['val_loss']
        }
        for school in ['A', 'B', 'C']
    }
}

with open(os.path.join(OUTPUT_PATH, 'training_histories.json'), 'w') as f:
    json.dump(histories_data, f, indent=2)
print("   Saved: training_histories.json")

# ================================================================================
# SCIENTIFIC INTERPRETATION
# ================================================================================

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

# Calculate statistics for interpretation
local_rmse_values = [local_metrics[s]['RMSE_ppm'] for s in ['A', 'B', 'C']]
local_r2_values = [local_metrics[s]['R2'] for s in ['A', 'B', 'C']]

avg_local_rmse = np.mean(local_rmse_values)
avg_local_r2 = np.mean(local_r2_values)
std_local_rmse = np.std(local_rmse_values)

best_local_school = ['A', 'B', 'C'][np.argmin(local_rmse_values)]
worst_local_school = ['A', 'B', 'C'][np.argmax(local_rmse_values)]

interpretation = f"""
================================================================================
SCIENTIFIC INTERPRETATION: BASELINE MODELING RESULTS
================================================================================

1. PERFORMANCE ACROSS SCHOOLS
-----------------------------
The local models show varying performance across schools:

   Best performing:  School {best_local_school} (RMSE: {local_metrics[best_local_school]['RMSE_ppm']:.2f} ppm)
   Worst performing: School {worst_local_school} (RMSE: {local_metrics[worst_local_school]['RMSE_ppm']:.2f} ppm)

   Performance variation (std RMSE): {std_local_rmse:.2f} ppm

This variation confirms the non-IID nature of the data, where each school
has distinct CO2 patterns that affect model performance differently.


2. CENTRALIZED VS LOCAL MODELS
------------------------------
Centralized Model:
   - RMSE: {centralized_metrics['RMSE_ppm']:.2f} ppm
   - R2:   {centralized_metrics['R2']:.4f}

Average Local Model:
   - RMSE: {avg_local_rmse:.2f} ppm
   - R2:   {avg_local_r2:.4f}

Comparison:
   - RMSE difference: {centralized_metrics['RMSE_ppm'] - avg_local_rmse:+.2f} ppm
     (positive = centralized is worse, negative = centralized is better)

   - When centralized model is tested per-school:
     School A: RMSE = {centralized_per_school['A']['RMSE_ppm']:.2f} ppm (vs local {local_metrics['A']['RMSE_ppm']:.2f} ppm)
     School B: RMSE = {centralized_per_school['B']['RMSE_ppm']:.2f} ppm (vs local {local_metrics['B']['RMSE_ppm']:.2f} ppm)
     School C: RMSE = {centralized_per_school['C']['RMSE_ppm']:.2f} ppm (vs local {local_metrics['C']['RMSE_ppm']:.2f} ppm)


3. SUITABILITY FOR FEDERATED LEARNING
-------------------------------------
The baseline results indicate this problem is well-suited for federated learning:

   a) Data Heterogeneity: The performance variation across local models
      ({std_local_rmse:.2f} ppm std in RMSE) demonstrates meaningful
      differences in data distributions between schools.

   b) Trade-off Opportunity: The centralized model benefits from more data
      but may not perfectly fit each school. Federated learning can
      potentially achieve a balance between:
      - Global knowledge sharing (like centralized)
      - Local adaptation (like local-only models)

   c) Privacy Motivation: Schools may not want to share raw CO2 data
      (reveals occupancy patterns, class schedules). Federated learning
      enables collaborative model training without data sharing.


4. BASELINE BENCHMARKS FOR FEDERATED COMPARISON
-----------------------------------------------
The following benchmarks are established:

   UPPER BOUND (ideal target):
   - Best local model performance on its own test set
   - Represents perfect local adaptation

   LOWER BOUND (minimum acceptable):
   - Centralized model performance
   - Represents what can be achieved with full data sharing

   FEDERATED GOAL:
   - Match or exceed centralized performance
   - Achieve this without centralizing raw data
   - Ideally approach local model performance per school


================================================================================
"""

print(interpretation)

# Save interpretation
with open(os.path.join(OUTPUT_PATH, 'baseline_interpretation.txt'), 'w') as f:
    f.write(interpretation)
print("Saved: baseline_interpretation.txt")

# ================================================================================
# FINAL SUMMARY TABLE
# ================================================================================

final_table = f"""
================================================================================
MISSION 2: BASELINE MODELING - FINAL SUMMARY
================================================================================

MODEL CONFIGURATION:
   - Architecture: LSTM({CONFIG['lstm_units']}) -> Dropout({CONFIG['dropout_rate']}) -> Dense(1)
   - Optimizer: Adam (lr={CONFIG['learning_rate']})
   - Batch Size: {CONFIG['batch_size']}
   - Max Epochs: {CONFIG['max_epochs']}
   - Early Stopping Patience: {CONFIG['patience']}

RESULTS TABLE:
+-------------------------+------------+------------+----------+
| Model                   | RMSE (ppm) | MAE (ppm)  | R2       |
+-------------------------+------------+------------+----------+
| Centralized (All)       | {centralized_metrics['RMSE_ppm']:>10.2f} | {centralized_metrics['MAE_ppm']:>10.2f} | {centralized_metrics['R2']:>8.4f} |
+-------------------------+------------+------------+----------+
| Local A                 | {local_metrics['A']['RMSE_ppm']:>10.2f} | {local_metrics['A']['MAE_ppm']:>10.2f} | {local_metrics['A']['R2']:>8.4f} |
| Local B                 | {local_metrics['B']['RMSE_ppm']:>10.2f} | {local_metrics['B']['MAE_ppm']:>10.2f} | {local_metrics['B']['R2']:>8.4f} |
| Local C                 | {local_metrics['C']['RMSE_ppm']:>10.2f} | {local_metrics['C']['MAE_ppm']:>10.2f} | {local_metrics['C']['R2']:>8.4f} |
+-------------------------+------------+------------+----------+
| Local Average           | {avg_local_rmse:>10.2f} | {np.mean([local_metrics[s]['MAE_ppm'] for s in ['A', 'B', 'C']]):>10.2f} | {avg_local_r2:>8.4f} |
+-------------------------+------------+------------+----------+

CENTRALIZED MODEL PER-SCHOOL PERFORMANCE:
+-------------------------+------------+------------+----------+
| Test Set                | RMSE (ppm) | MAE (ppm)  | R2       |
+-------------------------+------------+------------+----------+
| School A                | {centralized_per_school['A']['RMSE_ppm']:>10.2f} | {centralized_per_school['A']['MAE_ppm']:>10.2f} | {centralized_per_school['A']['R2']:>8.4f} |
| School B                | {centralized_per_school['B']['RMSE_ppm']:>10.2f} | {centralized_per_school['B']['MAE_ppm']:>10.2f} | {centralized_per_school['B']['R2']:>8.4f} |
| School C                | {centralized_per_school['C']['RMSE_ppm']:>10.2f} | {centralized_per_school['C']['MAE_ppm']:>10.2f} | {centralized_per_school['C']['R2']:>8.4f} |
+-------------------------+------------+------------+----------+

OUTPUT FILES:
   - baseline_results.json
   - training_histories.json
   - baseline_interpretation.txt
   - loss_curves.png
   - prediction_scatter.png
   - time_series_comparison.png
   - metrics_comparison.png
   - Centralized_LSTM_best.keras
   - Local_A_LSTM_best.keras
   - Local_B_LSTM_best.keras
   - Local_C_LSTM_best.keras

================================================================================
MISSION 2 COMPLETE
================================================================================
Baseline modeling finished. Ready for federated learning experiments.
Awaiting confirmation before proceeding to Mission 3.
================================================================================
"""

print(final_table)

with open(os.path.join(OUTPUT_PATH, 'mission2_summary.txt'), 'w') as f:
    f.write(final_table)
print("\nSaved: mission2_summary.txt")
