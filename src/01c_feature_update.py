"""
================================================================================
MISSION 1 UPDATE: REFINED FEATURE SELECTION
================================================================================
Research Project: Federated Learning for CO2 Prediction in Smart School Environments
Target: Q1 Journal Publication

This script updates the feature selection and window size based on refined requirements.

Changes from original:
- INPUT FEATURES: CO2 (lagged), Temperature, Humidity, hour_of_day, day_of_week
- IGNORED: PM2.5, PM10, CH2O, VOC
- WINDOW SIZE: 12 timesteps (increased from 6)

Author: Research Team
Date: March 2024
================================================================================
"""

import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler

# Paths
BASE_PATH = r"C:\Users\info\Documents\my-project\Federated Learning"
DATASET_PATH = os.path.join(BASE_PATH, "dataset-school")
OUTPUT_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis")

print("=" * 80)
print("MISSION 1 UPDATE: REFINED FEATURE SELECTION")
print("=" * 80)

# ================================================================================
# CONFIGURATION
# ================================================================================

# Define the exact features to use
FEATURE_CONFIG = {
    'target': 'CO2 (ppm)',
    'input_features': [
        'CO2 (ppm)',        # Lagged CO2 values
        'Temperature (°)',   # Environmental feature
        'Humidity (%)',      # Environmental feature
        'hour_of_day',       # Temporal feature (engineered)
        'day_of_week'        # Temporal feature (engineered)
    ],
    'ignored_features': [
        'PM2.5 (μg/m3)',     # Ignored per requirements
        'PM10 (μg/m3)',      # Ignored per requirements
        'CH2O (mg/m3)',      # Ignored per requirements
        'VOC (GRADE)'        # Ignored per requirements
    ]
}

WINDOW_SIZE = 12  # Updated from 6 to 12

print("\n" + "-" * 60)
print("FEATURE CONFIGURATION")
print("-" * 60)
print(f"\nTarget Variable: {FEATURE_CONFIG['target']}")
print(f"\nInput Features ({len(FEATURE_CONFIG['input_features'])}):")
for i, feat in enumerate(FEATURE_CONFIG['input_features'], 1):
    print(f"   {i}. {feat}")
print(f"\nIgnored Features ({len(FEATURE_CONFIG['ignored_features'])}):")
for feat in FEATURE_CONFIG['ignored_features']:
    print(f"   - {feat}")
print(f"\nWindow Size: {WINDOW_SIZE} timesteps")

# ================================================================================
# LOAD AND PREPARE DATA
# ================================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING CLEANED DATA")
print("=" * 80)

def load_and_prepare_school(school_name):
    """
    Load school data, apply cleaning, and extract required features.
    """
    file_path = os.path.join(DATASET_PATH, f"school-{school_name}.xlsx")
    df = pd.read_excel(file_path)

    print(f"\n   School {school_name}: Loaded {len(df):,} samples")

    # Convert timestamp
    df['time of read'] = pd.to_datetime(df['time of read'])

    # Extract temporal features
    df['hour_of_day'] = df['time of read'].dt.hour
    df['day_of_week'] = df['time of read'].dt.dayofweek

    # Apply same cleaning as before (spike detection and interpolation)
    co2_col = 'CO2 (ppm)'

    # Detect spikes (>300 ppm change)
    co2_diff = df[co2_col].diff().abs()
    mask_spike = co2_diff > 300
    n_spikes = mask_spike.sum()

    if n_spikes > 0:
        df.loc[mask_spike, co2_col] = np.nan
        # Also mark Temperature and Humidity as potentially affected
        df.loc[mask_spike, 'Temperature (°)'] = np.nan
        df.loc[mask_spike, 'Humidity (%)'] = np.nan

    # Interpolate small gaps
    for col in [co2_col, 'Temperature (°)', 'Humidity (%)']:
        df[col] = df[col].interpolate(method='linear', limit=3)

    # Drop any remaining NaN
    df = df.dropna(subset=[co2_col, 'Temperature (°)', 'Humidity (%)'])

    print(f"   School {school_name}: After cleaning: {len(df):,} samples")

    return df

# Load all schools
datasets = {}
for school in ['A', 'B', 'C']:
    datasets[school] = load_and_prepare_school(school)

# ================================================================================
# FEATURE EXTRACTION
# ================================================================================

print("\n" + "=" * 80)
print("STEP 2: EXTRACTING SELECTED FEATURES")
print("=" * 80)

# Map column names (handle special characters)
COLUMN_MAP = {
    'CO2 (ppm)': 'CO2 (ppm)',
    'Temperature (°)': 'Temperature (°)',
    'Humidity (%)': 'Humidity (%)',
    'hour_of_day': 'hour_of_day',
    'day_of_week': 'day_of_week'
}

feature_datasets = {}

for school in ['A', 'B', 'C']:
    df = datasets[school]

    # Extract only the selected features
    selected_cols = list(COLUMN_MAP.values())
    df_features = df[selected_cols].copy()

    feature_datasets[school] = df_features

    print(f"\n   School {school}:")
    print(f"   - Samples: {len(df_features):,}")
    print(f"   - Features: {list(df_features.columns)}")
    print(f"   - Shape: {df_features.shape}")

# ================================================================================
# SCALING (Fit on Combined Data)
# ================================================================================

print("\n" + "=" * 80)
print("STEP 3: SCALING FEATURES")
print("=" * 80)

print("\nFitting MinMaxScaler on combined data from all schools...")

# Combine all data for scaler fitting
combined_data = pd.concat([feature_datasets[s] for s in ['A', 'B', 'C']], ignore_index=True)
print(f"   Combined data shape: {combined_data.shape}")

# Fit scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(combined_data.values)

# Display learned ranges
feature_names = list(combined_data.columns)
print(f"\n   Feature Ranges Learned:")
print(f"   {'Feature':<20} {'Min':>12} {'Max':>12}")
print(f"   {'-'*20} {'-'*12} {'-'*12}")
for i, feat in enumerate(feature_names):
    print(f"   {feat:<20} {scaler.data_min_[i]:>12.2f} {scaler.data_max_[i]:>12.2f}")

# Apply scaling to each dataset
scaled_datasets = {}
for school in ['A', 'B', 'C']:
    df = feature_datasets[school]
    scaled_values = scaler.transform(df.values)
    scaled_datasets[school] = pd.DataFrame(scaled_values, columns=feature_names)
    print(f"\n   School {school} scaled: {scaled_datasets[school].shape}")

# Save updated scaler
with open(os.path.join(OUTPUT_PATH, "04_scaler_updated.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# ================================================================================
# SUPERVISED DATA PREPARATION (Window Size = 12)
# ================================================================================

print("\n" + "=" * 80)
print("STEP 4: SUPERVISED DATA PREPARATION")
print("=" * 80)

print(f"\nWindow Size: {WINDOW_SIZE} timesteps")
print(f"Features per timestep: {len(feature_names)}")
print(f"Total input dimensions: {WINDOW_SIZE} x {len(feature_names)} = {WINDOW_SIZE * len(feature_names)}")

"""
Data Structure Explanation:
===========================
For each sample at time t:

INPUT X[t]:
  - Shape: (window_size, n_features) = (12, 5)
  - Content: Features from timesteps [t-12, t-11, ..., t-1]
  - Each timestep contains: [CO2, Temperature, Humidity, hour_of_day, day_of_week]

TARGET y[t]:
  - Shape: scalar
  - Content: CO2 value at timestep t (normalized)

Example:
  X[t] = [[CO2(t-12), Temp(t-12), Hum(t-12), hour(t-12), day(t-12)],
          [CO2(t-11), Temp(t-11), Hum(t-11), hour(t-11), day(t-11)],
          ...
          [CO2(t-1),  Temp(t-1),  Hum(t-1),  hour(t-1),  day(t-1)]]

  y[t] = CO2(t)
"""

def create_sequences(data, target_col_idx, window_size):
    """
    Create input-output sequences for LSTM training.

    Parameters:
    -----------
    data : numpy.ndarray
        Scaled feature array (n_samples, n_features)
    target_col_idx : int
        Index of target column (CO2) in the feature array
    window_size : int
        Number of past timesteps to use as input

    Returns:
    --------
    X : numpy.ndarray
        Input sequences (n_sequences, window_size, n_features)
    y : numpy.ndarray
        Target values (n_sequences,)
    """
    X, y = [], []

    for i in range(window_size, len(data)):
        # Input: past window_size timesteps (all features)
        X.append(data[i-window_size:i, :])
        # Target: CO2 at current timestep
        y.append(data[i, target_col_idx])

    return np.array(X), np.array(y)

# Find CO2 column index
target_col_idx = feature_names.index('CO2 (ppm)')
print(f"\nTarget column index: {target_col_idx} ('{feature_names[target_col_idx]}')")

# Create sequences for each school
prepared_data = {}
preparation_summary = {}

print("\n" + "-" * 60)
print("Creating Sequences for Each School")
print("-" * 60)

for school in ['A', 'B', 'C']:
    df_scaled = scaled_datasets[school]
    data = df_scaled.values

    X, y = create_sequences(data, target_col_idx, WINDOW_SIZE)

    prepared_data[school] = {'X': X, 'y': y}

    preparation_summary[school] = {
        'n_original_samples': len(df_scaled),
        'n_sequences': len(X),
        'X_shape': list(X.shape),
        'y_shape': list(y.shape),
        'sequences_lost': WINDOW_SIZE  # First 'window_size' samples cannot form complete sequences
    }

    print(f"\n   School {school}:")
    print(f"   - Original samples: {len(df_scaled):,}")
    print(f"   - Sequences created: {len(X):,}")
    print(f"   - X shape: {X.shape}  (samples, timesteps, features)")
    print(f"   - y shape: {y.shape}  (samples,)")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY: UPDATED FEATURE CONFIGURATION")
print("=" * 80)

print("""
INPUT FEATURES (5 total):
=========================
1. CO2 (ppm)        - Target variable (lagged values as input)
2. Temperature (C)  - Environmental feature (affects CO2 dynamics)
3. Humidity (%)     - Environmental feature (affects CO2 dynamics)
4. hour_of_day      - Temporal feature (captures diurnal patterns)
5. day_of_week      - Temporal feature (captures weekly patterns)

IGNORED FEATURES:
=================
- PM2.5             - Not relevant for CO2 prediction
- PM10              - Not relevant for CO2 prediction
- CH2O              - Not relevant for CO2 prediction
- VOC               - Not relevant for CO2 prediction

JUSTIFICATION FOR FEATURE SELECTION:
====================================
1. CO2 (lagged): Essential for time-series prediction; captures temporal
   autocorrelation in CO2 levels.

2. Temperature: Indoor temperature affects metabolic CO2 production and
   ventilation behavior (window opening). Strong physical relationship.

3. Humidity: Correlates with occupancy patterns and ventilation. Higher
   humidity often indicates more people or less ventilation.

4. hour_of_day: Captures regular daily patterns in school occupancy
   (classes, breaks, after-school activities).

5. day_of_week: Distinguishes weekdays (high occupancy) from weekends
   (low/no occupancy) - critical for school environments.
""")

print("\nWINDOW SIZE JUSTIFICATION:")
print("=" * 40)
print(f"""
Window Size: {WINDOW_SIZE} timesteps (~12 minutes at 1-min sampling)

Rationale:
- Captures medium-term CO2 dynamics
- Sufficient for detecting accumulation trends
- Covers typical transition periods (class changes)
- Balances temporal context vs. computational cost
""")

print("\nUPDATED DATASET SHAPES:")
print("=" * 40)
print(f"{'School':<10} {'X Shape':<25} {'y Shape':<15} {'Samples':>10}")
print("-" * 60)
for school in ['A', 'B', 'C']:
    s = preparation_summary[school]
    print(f"{school:<10} {str(s['X_shape']):<25} {str(s['y_shape']):<15} {s['n_sequences']:>10,}")

total_samples = sum(s['n_sequences'] for s in preparation_summary.values())
print("-" * 60)
print(f"{'TOTAL':<10} {'':<25} {'':<15} {total_samples:>10,}")

print(f"""
DIMENSION BREAKDOWN:
====================
- X[i] shape: ({WINDOW_SIZE}, {len(feature_names)})
  - {WINDOW_SIZE} timesteps of historical data
  - {len(feature_names)} features per timestep

- y[i] shape: scalar
  - CO2 value at prediction timestep (normalized [0,1])

Feature Order in X:
  Index 0: CO2 (ppm)
  Index 1: Temperature (C)
  Index 2: Humidity (%)
  Index 3: hour_of_day
  Index 4: day_of_week
""")

# ================================================================================
# SAVE UPDATED DATA
# ================================================================================

print("\n" + "=" * 80)
print("SAVING UPDATED DATASETS")
print("=" * 80)

# Save numpy arrays
for school in ['A', 'B', 'C']:
    np.save(os.path.join(OUTPUT_PATH, f"05_X_school_{school}_v2.npy"), prepared_data[school]['X'])
    np.save(os.path.join(OUTPUT_PATH, f"05_y_school_{school}_v2.npy"), prepared_data[school]['y'])
    print(f"   Saved: 05_X_school_{school}_v2.npy, 05_y_school_{school}_v2.npy")

# Save configuration
config = {
    'version': 2,
    'feature_config': {
        'target': 'CO2 (ppm)',
        'input_features': feature_names,
        'n_features': len(feature_names),
        'ignored': FEATURE_CONFIG['ignored_features']
    },
    'window_size': WINDOW_SIZE,
    'scaler': {
        'type': 'MinMaxScaler',
        'range': [0, 1],
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist()
    },
    'data_summary': preparation_summary
}

with open(os.path.join(OUTPUT_PATH, "05_preparation_info_v2.json"), 'w') as f:
    json.dump(config, f, indent=2)
print(f"   Saved: 05_preparation_info_v2.json")

print("\n" + "=" * 80)
print("MISSION 1 UPDATE COMPLETE")
print("=" * 80)
print("\nData is ready for federated learning experiments.")
print("Awaiting confirmation before proceeding to Mission 2.")
