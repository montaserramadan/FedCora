"""
================================================================================
MISSION 1 FINAL CORRECTIONS: Q1 PUBLICATION STANDARDS
================================================================================
Research Project: Federated Learning for CO2 Prediction in Smart School Environments
Target: Q1 Journal Publication

CORRECTIONS APPLIED:
1. Temperature filtering: Keep only 15C <= Temperature <= 35C
2. Train/Validation/Test split: 70/15/15 (chronological, no shuffling)
3. Scaling: Fit ONLY on training data (no data leakage)
4. Inverse transform capability preserved

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
print("MISSION 1 FINAL CORRECTIONS: Q1 PUBLICATION STANDARDS")
print("=" * 80)

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    'features': ['CO2 (ppm)', 'Temperature (°)', 'Humidity (%)', 'hour_of_day', 'day_of_week'],
    'target': 'CO2 (ppm)',
    'target_idx': 0,
    'window_size': 12,
    'temperature_filter': {
        'min': 15.0,
        'max': 35.0
    },
    'split_ratios': {
        'train': 0.70,
        'val': 0.15,
        'test': 0.15
    },
    'spike_threshold': 300  # ppm
}

print("\n" + "-" * 60)
print("CONFIGURATION")
print("-" * 60)
print(f"Features: {CONFIG['features']}")
print(f"Window Size: {CONFIG['window_size']} timesteps")
print(f"Temperature Filter: [{CONFIG['temperature_filter']['min']}, {CONFIG['temperature_filter']['max']}] C")
print(f"Split Ratios: Train={CONFIG['split_ratios']['train']*100:.0f}%, "
      f"Val={CONFIG['split_ratios']['val']*100:.0f}%, "
      f"Test={CONFIG['split_ratios']['test']*100:.0f}%")

# ================================================================================
# STEP 1: LOAD AND FILTER DATA
# ================================================================================

print("\n" + "=" * 80)
print("STEP 1: LOAD DATA AND APPLY TEMPERATURE FILTERING")
print("=" * 80)

def load_and_filter_school(school_name):
    """
    Load school data, apply cleaning and temperature filtering.
    """
    file_path = os.path.join(DATASET_PATH, f"school-{school_name}.xlsx")
    df = pd.read_excel(file_path)

    original_size = len(df)
    print(f"\n   School {school_name}:")
    print(f"   - Original samples: {original_size:,}")

    # Convert timestamp and extract temporal features
    df['time of read'] = pd.to_datetime(df['time of read'])
    df['hour_of_day'] = df['time of read'].dt.hour
    df['day_of_week'] = df['time of read'].dt.dayofweek

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('time of read').reset_index(drop=True)

    # Apply CO2 spike detection and interpolation
    co2_col = 'CO2 (ppm)'
    temp_col = 'Temperature (°)'
    hum_col = 'Humidity (%)'

    # Detect CO2 spikes (>300 ppm change)
    co2_diff = df[co2_col].diff().abs()
    mask_spike = co2_diff > CONFIG['spike_threshold']
    n_spikes = mask_spike.sum()

    if n_spikes > 0:
        df.loc[mask_spike, co2_col] = np.nan
        print(f"   - CO2 spikes detected: {n_spikes}")

    # Interpolate small gaps (up to 3 consecutive)
    for col in [co2_col, temp_col, hum_col]:
        df[col] = df[col].interpolate(method='linear', limit=3)

    # Drop rows with remaining NaN in key columns
    df = df.dropna(subset=[co2_col, temp_col, hum_col])
    after_interpolation = len(df)

    # TEMPERATURE FILTERING (NEW)
    temp_min = CONFIG['temperature_filter']['min']
    temp_max = CONFIG['temperature_filter']['max']

    before_temp_filter = len(df)
    mask_temp_valid = (df[temp_col] >= temp_min) & (df[temp_col] <= temp_max)
    n_temp_invalid = (~mask_temp_valid).sum()

    # Report temperature values being filtered
    if n_temp_invalid > 0:
        invalid_temps = df.loc[~mask_temp_valid, temp_col]
        print(f"   - Temperature values outside [{temp_min}, {temp_max}] C: {n_temp_invalid}")
        print(f"     - Min invalid: {invalid_temps.min():.1f} C")
        print(f"     - Max invalid: {invalid_temps.max():.1f} C")

    df = df[mask_temp_valid].reset_index(drop=True)
    after_temp_filter = len(df)

    print(f"   - After temperature filtering: {after_temp_filter:,} samples")
    print(f"   - Data retention: {(after_temp_filter/original_size)*100:.2f}%")

    return df

# Load and filter all schools
datasets = {}
for school in ['A', 'B', 'C']:
    datasets[school] = load_and_filter_school(school)

# ================================================================================
# STEP 2: TRAIN/VALIDATION/TEST SPLIT (CHRONOLOGICAL)
# ================================================================================

print("\n" + "=" * 80)
print("STEP 2: CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT")
print("=" * 80)

print("""
IMPORTANT: Data Leakage Prevention
===================================
- Split is performed BEFORE scaling
- Each school is split independently to maintain temporal continuity
- NO shuffling - chronological order preserved
- Training: first 70% of each school's timeline
- Validation: next 15% of each school's timeline
- Testing: final 15% of each school's timeline
""")

def split_chronological(df, train_ratio=0.70, val_ratio=0.15):
    """
    Split dataframe chronologically into train/val/test sets.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe (already sorted chronologically)
    train_ratio : float
        Proportion for training (default 0.70)
    val_ratio : float
        Proportion for validation (default 0.15)

    Returns:
    --------
    train_df, val_df, test_df : pandas.DataFrame
        Split dataframes
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df

# Split each school
split_data = {}
split_summary = {}

print("\n" + "-" * 60)
print("Split Results per School")
print("-" * 60)

for school in ['A', 'B', 'C']:
    df = datasets[school]
    train_df, val_df, test_df = split_chronological(
        df,
        CONFIG['split_ratios']['train'],
        CONFIG['split_ratios']['val']
    )

    split_data[school] = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    split_summary[school] = {
        'total': len(df),
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df),
        'train_pct': len(train_df) / len(df) * 100,
        'val_pct': len(val_df) / len(df) * 100,
        'test_pct': len(test_df) / len(df) * 100
    }

    s = split_summary[school]
    print(f"\n   School {school}:")
    print(f"   - Total: {s['total']:,}")
    print(f"   - Train: {s['train']:,} ({s['train_pct']:.1f}%)")
    print(f"   - Val:   {s['val']:,} ({s['val_pct']:.1f}%)")
    print(f"   - Test:  {s['test']:,} ({s['test_pct']:.1f}%)")

# ================================================================================
# STEP 3: FIT SCALER ON TRAINING DATA ONLY (NO DATA LEAKAGE)
# ================================================================================

print("\n" + "=" * 80)
print("STEP 3: FIT SCALER ON TRAINING DATA ONLY")
print("=" * 80)

print("""
DATA LEAKAGE PREVENTION:
========================
The MinMaxScaler is fitted ONLY on the combined training data from all schools.
Validation and test sets are transformed using this scaler without refitting.
This ensures no information from future data leaks into the training process.
""")

# Extract features from training data only
feature_cols = CONFIG['features']

# Combine training data from all schools
train_combined = []
for school in ['A', 'B', 'C']:
    train_df = split_data[school]['train'][feature_cols]
    train_combined.append(train_df)

train_combined_df = pd.concat(train_combined, ignore_index=True)
print(f"Combined training data shape: {train_combined_df.shape}")

# Fit scaler on training data ONLY
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_combined_df.values)

# Display learned ranges (from training data only)
print(f"\nFeature Ranges (learned from TRAINING data only):")
print(f"{'Feature':<20} {'Train Min':>12} {'Train Max':>12}")
print(f"{'-'*20} {'-'*12} {'-'*12}")
for i, feat in enumerate(feature_cols):
    print(f"{feat:<20} {scaler.data_min_[i]:>12.2f} {scaler.data_max_[i]:>12.2f}")

# Save scaler parameters for inverse transformation
scaler_params = {
    'type': 'MinMaxScaler',
    'feature_range': [0, 1],
    'feature_names': feature_cols,
    'data_min': scaler.data_min_.tolist(),
    'data_max': scaler.data_max_.tolist(),
    'scale': scaler.scale_.tolist(),
    'fitted_on': 'training_data_only',
    'n_training_samples': len(train_combined_df)
}

# CO2-specific inverse transform parameters (for prediction evaluation)
co2_idx = feature_cols.index('CO2 (ppm)')
scaler_params['co2_inverse_transform'] = {
    'min_ppm': scaler.data_min_[co2_idx],
    'max_ppm': scaler.data_max_[co2_idx],
    'formula': 'CO2_ppm = normalized_value * (max - min) + min'
}

print(f"\nCO2 Inverse Transform Parameters:")
print(f"   - Min (training): {scaler_params['co2_inverse_transform']['min_ppm']:.2f} ppm")
print(f"   - Max (training): {scaler_params['co2_inverse_transform']['max_ppm']:.2f} ppm")
print(f"   - Formula: CO2_ppm = normalized * ({scaler_params['co2_inverse_transform']['max_ppm']:.2f} - {scaler_params['co2_inverse_transform']['min_ppm']:.2f}) + {scaler_params['co2_inverse_transform']['min_ppm']:.2f}")

# ================================================================================
# STEP 4: APPLY SCALING AND CREATE SEQUENCES
# ================================================================================

print("\n" + "=" * 80)
print("STEP 4: APPLY SCALING AND CREATE SEQUENCES")
print("=" * 80)

def create_sequences(data, target_idx, window_size):
    """
    Create input-output sequences for LSTM.

    Parameters:
    -----------
    data : numpy.ndarray
        Scaled feature array (n_samples, n_features)
    target_idx : int
        Index of target column (CO2)
    window_size : int
        Number of past timesteps

    Returns:
    --------
    X : numpy.ndarray
        Input sequences (n_sequences, window_size, n_features)
    y : numpy.ndarray
        Target values (n_sequences,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

# Process each school and each split
final_data = {}
final_summary = {}

window_size = CONFIG['window_size']
target_idx = CONFIG['target_idx']

print(f"\nWindow Size: {window_size} timesteps")
print(f"Target Index: {target_idx} ({feature_cols[target_idx]})")

for school in ['A', 'B', 'C']:
    print(f"\n" + "-" * 40)
    print(f"Processing School {school}")
    print("-" * 40)

    final_data[school] = {}
    final_summary[school] = {}

    for split_name in ['train', 'val', 'test']:
        # Get raw data
        df = split_data[school][split_name][feature_cols]

        # Apply scaling (using scaler fitted on training data)
        scaled_values = scaler.transform(df.values)

        # Create sequences
        X, y = create_sequences(scaled_values, target_idx, window_size)

        final_data[school][split_name] = {'X': X, 'y': y}
        final_summary[school][split_name] = {
            'raw_samples': len(df),
            'sequences': len(X),
            'X_shape': list(X.shape),
            'y_shape': list(y.shape)
        }

        print(f"   {split_name.upper():>5}: {len(df):,} samples -> {len(X):,} sequences, X: {X.shape}")

# ================================================================================
# STEP 5: VERIFICATION AND SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("STEP 5: FINAL VERIFICATION AND SUMMARY")
print("=" * 80)

# Aggregate statistics
total_train = sum(final_summary[s]['train']['sequences'] for s in ['A', 'B', 'C'])
total_val = sum(final_summary[s]['val']['sequences'] for s in ['A', 'B', 'C'])
total_test = sum(final_summary[s]['test']['sequences'] for s in ['A', 'B', 'C'])
total_all = total_train + total_val + total_test

print(f"""
DATASET SIZES AFTER SPLIT:
==========================

School A:
  - Train: {final_summary['A']['train']['sequences']:,} sequences
  - Val:   {final_summary['A']['val']['sequences']:,} sequences
  - Test:  {final_summary['A']['test']['sequences']:,} sequences

School B:
  - Train: {final_summary['B']['train']['sequences']:,} sequences
  - Val:   {final_summary['B']['val']['sequences']:,} sequences
  - Test:  {final_summary['B']['test']['sequences']:,} sequences

School C:
  - Train: {final_summary['C']['train']['sequences']:,} sequences
  - Val:   {final_summary['C']['val']['sequences']:,} sequences
  - Test:  {final_summary['C']['test']['sequences']:,} sequences

TOTALS (All Schools Combined):
  - Train: {total_train:,} sequences ({total_train/total_all*100:.1f}%)
  - Val:   {total_val:,} sequences ({total_val/total_all*100:.1f}%)
  - Test:  {total_test:,} sequences ({total_test/total_all*100:.1f}%)
  - Total: {total_all:,} sequences
""")

print(f"""
FINAL SHAPES:
=============

X Shape: (n_samples, {window_size}, {len(feature_cols)})
         (n_samples, timesteps, features)

y Shape: (n_samples,)
         (normalized CO2 values)

Feature Order:
  [0] CO2 (ppm)
  [1] Temperature (C)
  [2] Humidity (%)
  [3] hour_of_day
  [4] day_of_week
""")

print("""
DATA LEAKAGE VERIFICATION:
==========================
[OK] Chronological split applied (no shuffling)
[OK] Scaler fitted on TRAINING data only
[OK] Validation/Test scaled using training statistics
[OK] No future information leaked into training

INVERSE TRANSFORM FOR EVALUATION:
=================================
To convert normalized predictions back to ppm:

  CO2_ppm = y_normalized * (CO2_max - CO2_min) + CO2_min
""")
print(f"  CO2_ppm = y_normalized * ({scaler_params['co2_inverse_transform']['max_ppm']:.2f} - {scaler_params['co2_inverse_transform']['min_ppm']:.2f}) + {scaler_params['co2_inverse_transform']['min_ppm']:.2f}")

# ================================================================================
# STEP 6: SAVE ALL DATA
# ================================================================================

print("\n" + "=" * 80)
print("STEP 6: SAVING FINAL DATASETS")
print("=" * 80)

# Save data for each school and split
for school in ['A', 'B', 'C']:
    for split_name in ['train', 'val', 'test']:
        X = final_data[school][split_name]['X']
        y = final_data[school][split_name]['y']

        np.save(os.path.join(OUTPUT_PATH, f"final_X_{school}_{split_name}.npy"), X)
        np.save(os.path.join(OUTPUT_PATH, f"final_y_{school}_{split_name}.npy"), y)

print("   Saved: final_X_{A,B,C}_{train,val,test}.npy")
print("   Saved: final_y_{A,B,C}_{train,val,test}.npy")

# Save scaler
with open(os.path.join(OUTPUT_PATH, "final_scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)
print("   Saved: final_scaler.pkl")

# Save scaler parameters (for inverse transform)
with open(os.path.join(OUTPUT_PATH, "final_scaler_params.json"), 'w') as f:
    json.dump(scaler_params, f, indent=2)
print("   Saved: final_scaler_params.json")

# Save complete summary
complete_summary = {
    'config': {
        'features': CONFIG['features'],
        'window_size': CONFIG['window_size'],
        'target': CONFIG['target'],
        'temperature_filter': CONFIG['temperature_filter'],
        'split_ratios': CONFIG['split_ratios']
    },
    'data_summary': final_summary,
    'totals': {
        'train': total_train,
        'val': total_val,
        'test': total_test,
        'total': total_all
    },
    'scaler_params': scaler_params,
    'data_leakage_prevention': {
        'chronological_split': True,
        'scaler_fitted_on': 'training_data_only',
        'shuffle': False
    }
}

with open(os.path.join(OUTPUT_PATH, "final_data_summary.json"), 'w') as f:
    json.dump(complete_summary, f, indent=2)
print("   Saved: final_data_summary.json")

# ================================================================================
# FINAL REPORT
# ================================================================================

print("\n" + "=" * 80)
print("MISSION 1 FINAL CORRECTIONS COMPLETE")
print("=" * 80)

print("""
SUMMARY OF CORRECTIONS APPLIED:
===============================

1. TEMPERATURE FILTERING
   - Removed values outside [15, 35] C
   - Ensures realistic indoor school environment data

2. TRAIN/VALIDATION/TEST SPLIT
   - 70% Training / 15% Validation / 15% Testing
   - Chronological order preserved (no shuffling)
   - Each school split independently

3. SCALING (NO DATA LEAKAGE)
   - MinMaxScaler fitted on TRAINING data only
   - Same scaler applied to validation and test sets
   - No future information leaked

4. INVERSE TRANSFORM CAPABILITY
   - Scaler parameters saved for converting predictions to ppm
   - Essential for meaningful evaluation metrics (MAE, RMSE in ppm)

DATA IS READY FOR FEDERATED LEARNING EXPERIMENTS.
Awaiting confirmation before proceeding to Mission 2.
""")
