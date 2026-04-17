"""
================================================================================
MISSION 1: DATA INSPECTION AND PREPROCESSING FOR FEDERATED LEARNING
================================================================================
Research Project: Federated Learning for CO2 Prediction in Smart School Environments
Target: Q1 Journal Publication

This script performs comprehensive data preprocessing and non-IID analysis
for three federated clients (School A, B, C).

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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Paths
BASE_PATH = r"C:\Users\info\Documents\my-project\Federated Learning"
DATASET_PATH = os.path.join(BASE_PATH, "dataset-school")
OUTPUT_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis")

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ================================================================================
# SECTION 1: DATASET INSPECTION
# ================================================================================

print("=" * 80)
print("SECTION 1: DATASET INSPECTION")
print("=" * 80)

def load_and_inspect_dataset(file_path, school_name):
    """
    Load dataset and perform comprehensive inspection.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    school_name : str
        Name identifier for the school (A, B, or C)

    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    inspection_report : dict
        Dictionary containing inspection results
    """
    print(f"\n{'-' * 60}")
    print(f"Inspecting School {school_name}")
    print(f"{'-' * 60}")

    # Load data
    df = pd.read_excel(file_path)

    # Basic information
    n_samples = len(df)
    columns = list(df.columns)
    dtypes = df.dtypes.to_dict()

    print(f"\n1. Basic Information:")
    print(f"   - Number of samples: {n_samples:,}")
    print(f"   - Number of columns: {len(columns)}")
    print(f"   - Column names: {columns}")

    print(f"\n2. Data Types:")
    for col, dtype in dtypes.items():
        print(f"   - {col}: {dtype}")

    # Check for timestamp columns
    timestamp_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
            timestamp_cols.append(col)

    # Try to detect datetime columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100))
                timestamp_cols.append(col)
            except:
                pass

    timestamp_cols = list(set(timestamp_cols))
    print(f"\n3. Timestamp Detection:")
    print(f"   - Potential timestamp columns: {timestamp_cols if timestamp_cols else 'None detected'}")

    # Sampling interval analysis
    print(f"\n4. Sampling Interval Analysis:")
    if timestamp_cols:
        for ts_col in timestamp_cols:
            try:
                ts = pd.to_datetime(df[ts_col])
                diff = ts.diff().dropna()
                mode_interval = diff.mode()[0] if len(diff.mode()) > 0 else diff.median()
                print(f"   - Column '{ts_col}':")
                print(f"     - Most common interval: {mode_interval}")
                print(f"     - Min interval: {diff.min()}")
                print(f"     - Max interval: {diff.max()}")
            except Exception as e:
                print(f"   - Could not analyze '{ts_col}': {e}")
    else:
        # Check if index might be datetime
        if isinstance(df.index, pd.DatetimeIndex):
            diff = pd.Series(df.index).diff().dropna()
            print(f"   - Index-based timestamps detected")
            print(f"   - Most common interval: {diff.mode()[0]}")

    # Missing values
    print(f"\n5. Missing Values per Column:")
    missing = df.isnull().sum()
    for col in df.columns:
        pct = (missing[col] / n_samples) * 100
        print(f"   - {col}: {missing[col]:,} ({pct:.2f}%)")

    # CO2 Statistics
    print(f"\n6. CO2 Basic Statistics:")
    co2_col = None
    for col in df.columns:
        if 'co2' in col.lower():
            co2_col = col
            break

    if co2_col:
        co2_data = df[co2_col].dropna()
        stats = {
            'min': co2_data.min(),
            'max': co2_data.max(),
            'mean': co2_data.mean(),
            'std': co2_data.std(),
            'median': co2_data.median(),
            'q25': co2_data.quantile(0.25),
            'q75': co2_data.quantile(0.75)
        }
        print(f"   - Column used: '{co2_col}'")
        print(f"   - Min: {stats['min']:.2f} ppm")
        print(f"   - Max: {stats['max']:.2f} ppm")
        print(f"   - Mean: {stats['mean']:.2f} ppm")
        print(f"   - Std: {stats['std']:.2f} ppm")
        print(f"   - Median: {stats['median']:.2f} ppm")
        print(f"   - Q25: {stats['q25']:.2f} ppm")
        print(f"   - Q75: {stats['q75']:.2f} ppm")
    else:
        print("   - WARNING: No CO2 column found!")
        stats = None

    # Display first few rows
    print(f"\n7. First 5 rows:")
    print(df.head().to_string())

    inspection_report = {
        'school': school_name,
        'n_samples': n_samples,
        'columns': columns,
        'dtypes': {k: str(v) for k, v in dtypes.items()},
        'timestamp_columns': timestamp_cols,
        'missing_values': missing.to_dict(),
        'co2_column': co2_col,
        'co2_stats': stats
    }

    return df, inspection_report

# Load all datasets
datasets = {}
inspection_reports = {}

for school in ['A', 'B', 'C']:
    file_path = os.path.join(DATASET_PATH, f"school-{school}.xlsx")
    df, report = load_and_inspect_dataset(file_path, school)
    datasets[school] = df
    inspection_reports[school] = report

# Save inspection report
with open(os.path.join(OUTPUT_PATH, "01_inspection_report.json"), 'w') as f:
    json.dump(inspection_reports, f, indent=2, default=str)

print("\n" + "=" * 80)
print("Inspection reports saved to: 01_inspection_report.json")
print("=" * 80)


# ================================================================================
# SECTION 2: DATA CLEANING
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 2: DATA CLEANING")
print("=" * 80)

# Define cleaning parameters (SAME for all datasets - critical for fairness)
CLEANING_PARAMS = {
    'co2_min': 350,          # Minimum physically plausible CO2 (ppm)
    'co2_max': 5000,         # Maximum physically plausible CO2 (ppm)
    'spike_threshold': 300,   # Maximum allowed CO2 change between consecutive samples (ppm)
    'small_gap_max': 3,       # Maximum consecutive missing values to interpolate
    'large_gap_min': 4        # Minimum consecutive missing values to remove segment
}

print("\nCleaning Parameters (Applied Consistently Across All Schools):")
print(f"  - CO2 valid range: [{CLEANING_PARAMS['co2_min']}, {CLEANING_PARAMS['co2_max']}] ppm")
print(f"  - Spike threshold: {CLEANING_PARAMS['spike_threshold']} ppm per timestep")
print(f"  - Small gap (interpolate): <= {CLEANING_PARAMS['small_gap_max']} consecutive missing values")
print(f"  - Large gap (remove): >= {CLEANING_PARAMS['large_gap_min']} consecutive missing values")

def clean_dataset(df, school_name, params):
    """
    Clean dataset following rigorous and consistent rules.

    Cleaning Steps:
    1. Remove physically impossible CO2 values
    2. Detect and handle abnormal spikes
    3. Handle missing values (interpolate small gaps, remove large gaps)

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    school_name : str
        School identifier
    params : dict
        Cleaning parameters

    Returns:
    --------
    df_cleaned : pandas.DataFrame
        Cleaned dataframe
    cleaning_log : dict
        Detailed log of cleaning operations
    """
    print(f"\n{'-' * 60}")
    print(f"Cleaning School {school_name}")
    print(f"{'-' * 60}")

    df_clean = df.copy()
    original_size = len(df_clean)
    cleaning_log = {'school': school_name, 'original_size': original_size}

    # Identify CO2 column
    co2_col = None
    for col in df_clean.columns:
        if 'co2' in col.lower():
            co2_col = col
            break

    if co2_col is None:
        print("ERROR: No CO2 column found!")
        return df_clean, cleaning_log

    cleaning_log['co2_column'] = co2_col

    # Step 2A: Remove physically impossible values
    print(f"\nStep 2A: Removing physically impossible CO2 values")
    mask_invalid = (df_clean[co2_col] < params['co2_min']) | (df_clean[co2_col] > params['co2_max'])
    n_invalid = mask_invalid.sum()

    if n_invalid > 0:
        invalid_values = df_clean.loc[mask_invalid, co2_col].values
        print(f"   - Found {n_invalid} invalid values")
        print(f"   - Examples: {invalid_values[:5]}")
        # Replace with NaN for later interpolation
        df_clean.loc[mask_invalid, co2_col] = np.nan
    else:
        print(f"   - No invalid values found")

    cleaning_log['invalid_co2_removed'] = int(n_invalid)

    # Step 2B: Detect and handle abnormal spikes
    print(f"\nStep 2B: Detecting abnormal spikes (threshold: {params['spike_threshold']} ppm)")
    co2_diff = df_clean[co2_col].diff().abs()
    mask_spike = co2_diff > params['spike_threshold']
    n_spikes = mask_spike.sum()

    if n_spikes > 0:
        print(f"   - Found {n_spikes} spike anomalies")
        spike_values = co2_diff[mask_spike].values[:5]
        print(f"   - Example jumps: {spike_values}")
        # Strategy: Replace spike values with NaN (will be interpolated or removed)
        # We mark the value AFTER the spike as anomalous
        df_clean.loc[mask_spike, co2_col] = np.nan
        print(f"   - Approach: Marked spike values as NaN for interpolation")
    else:
        print(f"   - No spikes detected")

    cleaning_log['spikes_detected'] = int(n_spikes)

    # Step 2C: Handle missing values
    print(f"\nStep 2C: Handling missing values")

    # Count current NaN values
    n_missing = df_clean[co2_col].isna().sum()
    print(f"   - Total missing values (including marked anomalies): {n_missing}")

    # Identify gap sizes
    is_nan = df_clean[co2_col].isna()
    gap_groups = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(gap_groups).transform('sum')

    # Small gaps: interpolate
    small_gap_mask = is_nan & (gap_sizes <= params['small_gap_max'])
    n_interpolated = small_gap_mask.sum()

    # Large gaps: mark for removal
    large_gap_mask = is_nan & (gap_sizes >= params['large_gap_min'])
    n_large_gaps = large_gap_mask.sum()

    print(f"   - Small gaps (<={params['small_gap_max']} consecutive): {n_interpolated} values → will interpolate")
    print(f"   - Large gaps (>={params['large_gap_min']} consecutive): {n_large_gaps} values → will remove")

    # Interpolate small gaps (linear interpolation)
    df_clean[co2_col] = df_clean[co2_col].interpolate(method='linear', limit=params['small_gap_max'])

    # Remove rows with large gaps (still NaN after interpolation)
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(subset=[co2_col])
    rows_removed = rows_before - len(df_clean)

    print(f"   - Rows removed due to large gaps: {rows_removed}")

    cleaning_log['values_interpolated'] = int(n_interpolated)
    cleaning_log['rows_removed_large_gaps'] = int(rows_removed)
    cleaning_log['final_size'] = len(df_clean)
    cleaning_log['data_retention_rate'] = len(df_clean) / original_size * 100

    print(f"\nCleaning Summary for School {school_name}:")
    print(f"   - Original size: {original_size:,} samples")
    print(f"   - Final size: {len(df_clean):,} samples")
    print(f"   - Data retention rate: {cleaning_log['data_retention_rate']:.2f}%")

    return df_clean, cleaning_log

# Apply cleaning to all datasets
cleaned_datasets = {}
cleaning_logs = {}

for school in ['A', 'B', 'C']:
    df_cleaned, log = clean_dataset(datasets[school], school, CLEANING_PARAMS)
    cleaned_datasets[school] = df_cleaned
    cleaning_logs[school] = log

# Save cleaning logs
with open(os.path.join(OUTPUT_PATH, "02_cleaning_log.json"), 'w') as f:
    json.dump(cleaning_logs, f, indent=2)

print("\n" + "=" * 80)
print("Cleaning logs saved to: 02_cleaning_log.json")
print("=" * 80)


# ================================================================================
# SECTION 3: FEATURE SELECTION AND ENGINEERING
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 3: FEATURE SELECTION AND ENGINEERING")
print("=" * 80)

def engineer_features(df, school_name):
    """
    Extract temporal features from timestamp data.

    Features extracted:
    - hour_of_day: Captures diurnal patterns in CO2 (occupancy varies throughout the day)
    - day_of_week: Captures weekly patterns (weekday vs weekend occupancy differences)

    Scientific Justification:
    - CO2 levels in schools are primarily driven by human occupancy and ventilation
    - Hour of day correlates with class schedules (morning arrival, lunch breaks, dismissal)
    - Day of week captures weekend patterns when schools are typically unoccupied

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe
    school_name : str
        School identifier

    Returns:
    --------
    df_featured : pandas.DataFrame
        Dataframe with engineered features
    feature_info : dict
        Information about extracted features
    """
    print(f"\n{'-' * 60}")
    print(f"Feature Engineering for School {school_name}")
    print(f"{'-' * 60}")

    df_feat = df.copy()
    feature_info = {'school': school_name, 'features_added': []}

    # Identify timestamp column
    timestamp_col = None
    for col in df_feat.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            timestamp_col = col
            break
        if df_feat[col].dtype == 'datetime64[ns]':
            timestamp_col = col
            break

    # Try to convert potential datetime columns
    if timestamp_col is None:
        for col in df_feat.columns:
            try:
                df_feat[col] = pd.to_datetime(df_feat[col])
                if df_feat[col].dtype == 'datetime64[ns]':
                    timestamp_col = col
                    break
            except:
                pass

    if timestamp_col:
        print(f"   - Timestamp column identified: '{timestamp_col}'")

        # Ensure datetime type
        df_feat[timestamp_col] = pd.to_datetime(df_feat[timestamp_col])

        # Extract hour of day (0-23)
        df_feat['hour_of_day'] = df_feat[timestamp_col].dt.hour
        feature_info['features_added'].append('hour_of_day')
        print(f"   - Extracted: hour_of_day (0-23)")

        # Extract day of week (0=Monday, 6=Sunday)
        df_feat['day_of_week'] = df_feat[timestamp_col].dt.dayofweek
        feature_info['features_added'].append('day_of_week')
        print(f"   - Extracted: day_of_week (0=Monday, 6=Sunday)")

        # Store timestamp column name
        feature_info['timestamp_column'] = timestamp_col

        print(f"\n   Scientific Justification:")
        print(f"   - hour_of_day: Captures diurnal CO2 patterns related to")
        print(f"     class schedules (arrival, breaks, dismissal)")
        print(f"   - day_of_week: Captures weekly patterns")
        print(f"     (occupied weekdays vs. unoccupied weekends)")
    else:
        print("   - WARNING: No timestamp column found")
        print("   - Creating index-based features as fallback")
        df_feat['time_index'] = np.arange(len(df_feat))
        feature_info['features_added'].append('time_index')

    # Identify CO2 column
    co2_col = None
    for col in df_feat.columns:
        if 'co2' in col.lower():
            co2_col = col
            break

    feature_info['co2_column'] = co2_col
    feature_info['final_columns'] = list(df_feat.columns)

    print(f"\n   Final columns: {list(df_feat.columns)}")

    return df_feat, feature_info

# Apply feature engineering to all datasets
featured_datasets = {}
feature_infos = {}

for school in ['A', 'B', 'C']:
    df_featured, info = engineer_features(cleaned_datasets[school], school)
    featured_datasets[school] = df_featured
    feature_infos[school] = info

# Save feature info
with open(os.path.join(OUTPUT_PATH, "03_feature_engineering_info.json"), 'w') as f:
    json.dump(feature_infos, f, indent=2)

print("\n" + "=" * 80)
print("Feature engineering info saved to: 03_feature_engineering_info.json")
print("=" * 80)


# ================================================================================
# SECTION 4: SCALING
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 4: SCALING")
print("=" * 80)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""
Scaler Selection Justification:
===============================

We choose MinMaxScaler for the following reasons:

1. LSTM Compatibility:
   - LSTM networks with sigmoid/tanh activations work best with bounded inputs [0, 1]
   - MinMaxScaler guarantees bounded output range

2. CO2 Data Characteristics:
   - CO2 has natural physical bounds (350-5000 ppm after cleaning)
   - MinMaxScaler preserves the relative relationships within the bounded range

3. Feature Consistency:
   - hour_of_day (0-23) and day_of_week (0-6) are naturally bounded
   - MinMaxScaler handles these consistently with CO2

4. Interpretability:
   - Scaled values can be easily inverse-transformed for prediction interpretation

Note: We fit the scaler on the COMBINED data from all schools to ensure
consistent scaling across the federated setting. This simulates a scenario
where global statistics are known or estimated.
"""

print("\nScaler Selection: MinMaxScaler")
print("\nJustification:")
print("  1. LSTM Compatibility: Bounded [0,1] range optimal for sigmoid/tanh activations")
print("  2. CO2 Physical Bounds: Preserves relationships within natural 350-5000 ppm range")
print("  3. Feature Consistency: Handles temporal features (hour, day) uniformly")
print("  4. Interpretability: Easy inverse transformation for prediction analysis")

# Identify columns to scale
# We need CO2 and the engineered features
def get_feature_columns(df):
    """Get list of feature columns for scaling."""
    cols = []
    for col in df.columns:
        if 'co2' in col.lower():
            cols.append(col)
        elif col in ['hour_of_day', 'day_of_week']:
            cols.append(col)
    return cols

# Combine all data to fit scaler (ensures consistent scaling)
print("\n" + "─" * 60)
print("Fitting scaler on combined data from all schools")
print("─" * 60)

combined_data = []
feature_cols = None

for school in ['A', 'B', 'C']:
    df = featured_datasets[school]
    cols = get_feature_columns(df)
    if feature_cols is None:
        feature_cols = cols
    combined_data.append(df[cols].values)
    print(f"   School {school}: {len(df):,} samples, features: {cols}")

combined_array = np.vstack(combined_data)
print(f"\n   Combined data shape: {combined_array.shape}")

# Fit MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(combined_array)

print(f"\n   Scaler fitted on {len(combined_array):,} total samples")
print(f"   Feature ranges learned:")
for i, col in enumerate(feature_cols):
    print(f"     - {col}: [{scaler.data_min_[i]:.2f}, {scaler.data_max_[i]:.2f}]")

# Apply scaling to each dataset
scaled_datasets = {}
scaling_info = {
    'scaler_type': 'MinMaxScaler',
    'feature_range': [0, 1],
    'feature_columns': feature_cols,
    'data_min': scaler.data_min_.tolist(),
    'data_max': scaler.data_max_.tolist(),
    'schools': {}
}

for school in ['A', 'B', 'C']:
    df = featured_datasets[school].copy()
    df_scaled = df.copy()

    # Scale the feature columns
    df_scaled[feature_cols] = scaler.transform(df[feature_cols].values)
    scaled_datasets[school] = df_scaled

    # Record scaling info
    scaling_info['schools'][school] = {
        'n_samples': len(df_scaled),
        'scaled_stats': {
            col: {
                'min': float(df_scaled[col].min()),
                'max': float(df_scaled[col].max()),
                'mean': float(df_scaled[col].mean()),
                'std': float(df_scaled[col].std())
            } for col in feature_cols
        }
    }

    print(f"\n   School {school} scaled: {len(df_scaled):,} samples")

# Save scaler and scaling info
import pickle
with open(os.path.join(OUTPUT_PATH, "04_scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(OUTPUT_PATH, "04_scaling_info.json"), 'w') as f:
    json.dump(scaling_info, f, indent=2)

print("\n" + "=" * 80)
print("Scaler saved to: 04_scaler.pkl")
print("Scaling info saved to: 04_scaling_info.json")
print("=" * 80)


# ================================================================================
# SECTION 5: SUPERVISED DATA PREPARATION
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 5: SUPERVISED DATA PREPARATION (LSTM-Ready)")
print("=" * 80)

"""
Window Size Selection: 6 timesteps
==================================

Justification:
1. Temporal Context:
   - If sampling is every 5-10 minutes, 6 timesteps ~ 30-60 minutes of history
   - This captures short-term CO2 dynamics (accumulation/ventilation effects)

2. CO2 Dynamics:
   - CO2 changes in indoor environments are relatively gradual
   - 6 timesteps provide sufficient context for trend detection

3. Model Complexity:
   - Smaller window = faster training, less overfitting risk
   - 6 is a reasonable balance between context and complexity

4. Literature Alignment:
   - Common practice in indoor air quality prediction
   - Allows model to learn both immediate changes and short-term trends
"""

WINDOW_SIZE = 6  # Number of past timesteps to use as input

print(f"\nWindow Size: {WINDOW_SIZE} timesteps")
print("\nJustification:")
print("  1. Temporal Context: Captures 30-60 minutes of history (at 5-10 min sampling)")
print("  2. CO2 Dynamics: Sufficient for detecting accumulation/ventilation patterns")
print("  3. Model Complexity: Balanced between context richness and overfitting risk")
print("  4. Literature Alignment: Standard practice for indoor air quality prediction")

def create_sequences(data, co2_col_idx, window_size):
    """
    Create sequences for LSTM training.

    Input format:
    X[t] = [features(t-window_size), features(t-window_size+1), ..., features(t-1)]
    y[t] = CO2(t)

    Parameters:
    -----------
    data : numpy.ndarray
        Scaled feature array (n_samples, n_features)
    co2_col_idx : int
        Index of CO2 column in the feature array
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
        # Input: past window_size timesteps (all features)
        X.append(data[i-window_size:i, :])
        # Target: CO2 at current timestep
        y.append(data[i, co2_col_idx])

    return np.array(X), np.array(y)

# Find CO2 column index
co2_col_idx = feature_cols.index([c for c in feature_cols if 'co2' in c.lower()][0])
print(f"\nCO2 column index in feature array: {co2_col_idx}")

# Create sequences for each school
prepared_data = {}
preparation_info = {
    'window_size': WINDOW_SIZE,
    'feature_columns': feature_cols,
    'co2_column_index': co2_col_idx,
    'schools': {}
}

print(f"\n{'─' * 60}")
print("Creating sequences for each school")
print("─" * 60)

for school in ['A', 'B', 'C']:
    df = scaled_datasets[school]
    data = df[feature_cols].values

    X, y = create_sequences(data, co2_col_idx, WINDOW_SIZE)

    prepared_data[school] = {'X': X, 'y': y}

    preparation_info['schools'][school] = {
        'X_shape': list(X.shape),
        'y_shape': list(y.shape),
        'n_sequences': len(X)
    }

    print(f"\n   School {school}:")
    print(f"     - X shape: {X.shape} (samples, timesteps, features)")
    print(f"     - y shape: {y.shape} (samples,)")
    print(f"     - Number of sequences: {len(X):,}")

# Save prepared data
for school in ['A', 'B', 'C']:
    np.save(os.path.join(OUTPUT_PATH, f"05_X_school_{school}.npy"), prepared_data[school]['X'])
    np.save(os.path.join(OUTPUT_PATH, f"05_y_school_{school}.npy"), prepared_data[school]['y'])

with open(os.path.join(OUTPUT_PATH, "05_preparation_info.json"), 'w') as f:
    json.dump(preparation_info, f, indent=2)

print("\n" + "=" * 80)
print("Prepared data saved:")
print("  - 05_X_school_A.npy, 05_y_school_A.npy")
print("  - 05_X_school_B.npy, 05_y_school_B.npy")
print("  - 05_X_school_C.npy, 05_y_school_C.npy")
print("  - 05_preparation_info.json")
print("=" * 80)


# ================================================================================
# SECTION 6: NON-IID ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 6: NON-IID ANALYSIS (Critical for Federated Learning)")
print("=" * 80)

# Get original (unscaled) CO2 values for meaningful analysis
co2_col_name = [c for c in feature_cols if 'co2' in c.lower()][0]

non_iid_stats = {}

print("\n" + "─" * 60)
print("Computing CO2 Statistics per School")
print("─" * 60)

for school in ['A', 'B', 'C']:
    df = featured_datasets[school]  # Use unscaled for interpretable statistics
    co2_values = df[co2_col_name].values

    stats = {
        'n_samples': len(co2_values),
        'mean': float(np.mean(co2_values)),
        'std': float(np.std(co2_values)),
        'min': float(np.min(co2_values)),
        'max': float(np.max(co2_values)),
        'median': float(np.median(co2_values)),
        'q25': float(np.percentile(co2_values, 25)),
        'q75': float(np.percentile(co2_values, 75)),
        'skewness': float(pd.Series(co2_values).skew()),
        'kurtosis': float(pd.Series(co2_values).kurtosis())
    }

    non_iid_stats[school] = stats

    print(f"\n   School {school}:")
    print(f"     - Samples: {stats['n_samples']:,}")
    print(f"     - Mean: {stats['mean']:.2f} ppm")
    print(f"     - Std: {stats['std']:.2f} ppm")
    print(f"     - Range: [{stats['min']:.2f}, {stats['max']:.2f}] ppm")
    print(f"     - Median: {stats['median']:.2f} ppm")
    print(f"     - IQR: [{stats['q25']:.2f}, {stats['q75']:.2f}] ppm")
    print(f"     - Skewness: {stats['skewness']:.3f}")
    print(f"     - Kurtosis: {stats['kurtosis']:.3f}")

# Save statistics
with open(os.path.join(OUTPUT_PATH, "06_non_iid_statistics.json"), 'w') as f:
    json.dump(non_iid_stats, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 1: CO2 Histograms for Each School
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("Creating Visualization 1: CO2 Histograms")
print("─" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#e74c3c'}

for idx, school in enumerate(['A', 'B', 'C']):
    ax = axes[idx]
    df = featured_datasets[school]
    co2_values = df[co2_col_name].values

    ax.hist(co2_values, bins=50, color=colors[school], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(co2_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(co2_values):.0f}')
    ax.axvline(np.median(co2_values), color='blue', linestyle=':', linewidth=2, label=f'Median: {np.median(co2_values):.0f}')

    ax.set_xlabel('CO2 Concentration (ppm)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'School {school}\n(n={len(co2_values):,}, std={np.std(co2_values):.0f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('CO2 Distribution by School (Non-IID Analysis)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "06_co2_histograms.png"), dpi=300, bbox_inches='tight')
plt.close()

print("   Saved: 06_co2_histograms.png")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 2: CO2 Boxplot Comparison
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("Creating Visualization 2: CO2 Boxplot Comparison")
print("─" * 60)

fig, ax = plt.subplots(figsize=(10, 7))

boxplot_data = []
boxplot_labels = []

for school in ['A', 'B', 'C']:
    df = featured_datasets[school]
    boxplot_data.append(df[co2_col_name].values)
    boxplot_labels.append(f'School {school}')

bp = ax.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)

# Color the boxes
for patch, school in zip(bp['boxes'], ['A', 'B', 'C']):
    patch.set_facecolor(colors[school])
    patch.set_alpha(0.7)

# Add mean markers
means = [np.mean(d) for d in boxplot_data]
ax.scatter([1, 2, 3], means, color='red', marker='D', s=100, zorder=5, label='Mean')

ax.set_ylabel('CO2 Concentration (ppm)')
ax.set_title('CO2 Distribution Comparison Across Schools\n(Boxplot Analysis for Non-IID Characterization)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper right')

# Add statistics annotation
stats_text = "Statistics Summary:\n"
for school in ['A', 'B', 'C']:
    s = non_iid_stats[school]
    stats_text += f"School {school}: mean={s['mean']:.0f}, std={s['std']:.0f}\n"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "06_co2_boxplot.png"), dpi=300, bbox_inches='tight')
plt.close()

print("   Saved: 06_co2_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 3: Combined Distribution Plot (KDE)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("Creating Visualization 3: Kernel Density Estimation")
print("─" * 60)

fig, ax = plt.subplots(figsize=(12, 6))

for school in ['A', 'B', 'C']:
    df = featured_datasets[school]
    co2_values = df[co2_col_name].values

    sns.kdeplot(co2_values, ax=ax, label=f'School {school} (n={len(co2_values):,})',
                color=colors[school], linewidth=2, fill=True, alpha=0.3)

ax.set_xlabel('CO2 Concentration (ppm)')
ax.set_ylabel('Density')
ax.set_title('CO2 Distribution Density Comparison (KDE)\nDemonstrating Non-IID Data Heterogeneity', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "06_co2_kde_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

print("   Saved: 06_co2_kde_comparison.png")


# ================================================================================
# SECTION 7: SCIENTIFIC INTERPRETATION AND SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("SECTION 7: SCIENTIFIC INTERPRETATION")
print("=" * 80)

# Calculate comparative metrics
mean_values = {s: non_iid_stats[s]['mean'] for s in ['A', 'B', 'C']}
std_values = {s: non_iid_stats[s]['std'] for s in ['A', 'B', 'C']}

highest_mean_school = max(mean_values, key=mean_values.get)
highest_std_school = max(std_values, key=std_values.get)
lowest_mean_school = min(mean_values, key=mean_values.get)

# Calculate coefficient of variation (CV) for each school
cv_values = {s: (non_iid_stats[s]['std'] / non_iid_stats[s]['mean']) * 100 for s in ['A', 'B', 'C']}

interpretation = f"""
================================================================================
SCIENTIFIC INTERPRETATION: Non-IID Analysis Results
================================================================================

1. DISTRIBUTION HETEROGENEITY
-----------------------------
The three schools exhibit distinct CO2 distributions, confirming the non-IID
nature of the federated learning scenario:

   School A: Mean = {non_iid_stats['A']['mean']:.2f} ppm, Std = {non_iid_stats['A']['std']:.2f} ppm
   School B: Mean = {non_iid_stats['B']['mean']:.2f} ppm, Std = {non_iid_stats['B']['std']:.2f} ppm
   School C: Mean = {non_iid_stats['C']['mean']:.2f} ppm, Std = {non_iid_stats['C']['std']:.2f} ppm

2. KEY OBSERVATIONS
-------------------
   • Highest Average CO2: School {highest_mean_school} ({mean_values[highest_mean_school]:.2f} ppm)
     - Possible causes: Higher occupancy, poorer ventilation, or different usage patterns

   • Highest Variability: School {highest_std_school} ({std_values[highest_std_school]:.2f} ppm std)
     - Indicates more dynamic CO2 patterns (varying occupancy or ventilation)

   • Coefficient of Variation:
     - School A: {cv_values['A']:.2f}%
     - School B: {cv_values['B']:.2f}%
     - School C: {cv_values['C']:.2f}%

3. IMPLICATIONS FOR FEDERATED LEARNING
--------------------------------------
   • Data Heterogeneity Challenge:
     The significant differences in mean and variance across schools create a
     non-IID scenario that challenges standard FedAvg convergence.

   • Statistical Divergence:
     Schools with higher variability may dominate gradient updates if not
     properly weighted, leading to biased global models.

   • Personalization Need:
     The distinct distributions suggest that personalized or adaptive federated
     learning approaches (e.g., FedProx, Per-FedAvg) may outperform vanilla FedAvg.

   • Local Optima Risk:
     Each school's local minimum differs from the global optimum, necessitating
     careful aggregation strategies.

4. SAMPLE SIZE DISTRIBUTION
---------------------------
   School A: {non_iid_stats['A']['n_samples']:,} samples ({non_iid_stats['A']['n_samples'] / sum(s['n_samples'] for s in non_iid_stats.values()) * 100:.1f}%)
   School B: {non_iid_stats['B']['n_samples']:,} samples ({non_iid_stats['B']['n_samples'] / sum(s['n_samples'] for s in non_iid_stats.values()) * 100:.1f}%)
   School C: {non_iid_stats['C']['n_samples']:,} samples ({non_iid_stats['C']['n_samples'] / sum(s['n_samples'] for s in non_iid_stats.values()) * 100:.1f}%)

   Note: Unequal sample sizes further contribute to the non-IID characteristics
   and require consideration in aggregation weighting.

================================================================================
"""

print(interpretation)

# Save interpretation
with open(os.path.join(OUTPUT_PATH, "07_scientific_interpretation.txt"), 'w') as f:
    f.write(interpretation)

print("Interpretation saved to: 07_scientific_interpretation.txt")


# ================================================================================
# FINAL SUMMARY REPORT
# ================================================================================

final_summary = f"""
================================================================================
MISSION 1: DATA INSPECTION AND PREPROCESSING - FINAL SUMMARY
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project: Federated Learning for CO2 Prediction in Smart School Environments
================================================================================

SECTION 1: DATASET INSPECTION
-----------------------------
Three datasets were inspected (School A, B, C) for:
• Sample counts, column names, data types
• Timestamp presence and sampling intervals
• Missing values per column
• CO2 basic statistics

SECTION 2: DATA CLEANING
------------------------
Consistent cleaning rules applied across all schools:
• CO2 valid range: [{CLEANING_PARAMS['co2_min']}, {CLEANING_PARAMS['co2_max']}] ppm
• Spike detection threshold: {CLEANING_PARAMS['spike_threshold']} ppm per timestep
• Small gaps (<={CLEANING_PARAMS['small_gap_max']} values): Linear interpolation
• Large gaps (>={CLEANING_PARAMS['large_gap_min']} values): Segment removal

Data Retention Rates:
• School A: {cleaning_logs['A']['data_retention_rate']:.2f}%
• School B: {cleaning_logs['B']['data_retention_rate']:.2f}%
• School C: {cleaning_logs['C']['data_retention_rate']:.2f}%

SECTION 3: FEATURE ENGINEERING
------------------------------
Temporal features extracted from timestamps:
• hour_of_day (0-23): Captures diurnal occupancy patterns
• day_of_week (0-6): Captures weekly patterns (weekday vs weekend)

SECTION 4: SCALING
------------------
Scaler: MinMaxScaler (range [0, 1])
Justification: LSTM compatibility, bounded CO2 range, consistent feature treatment
Applied: Fitted on combined data from all schools for consistency

SECTION 5: SUPERVISED DATA PREPARATION
--------------------------------------
Window size: {WINDOW_SIZE} timesteps
Format: X[t] = [features(t-{WINDOW_SIZE}:t-1)], y[t] = CO2(t)

Final Dataset Shapes:
• School A: X = {preparation_info['schools']['A']['X_shape']}, y = {preparation_info['schools']['A']['y_shape']}
• School B: X = {preparation_info['schools']['B']['X_shape']}, y = {preparation_info['schools']['B']['y_shape']}
• School C: X = {preparation_info['schools']['C']['X_shape']}, y = {preparation_info['schools']['C']['y_shape']}

SECTION 6: NON-IID ANALYSIS
---------------------------
CO2 Statistics (demonstrating heterogeneity):

School  |   Mean   |   Std    |   Min    |   Max    | Samples
--------|----------|----------|----------|----------|--------
   A    | {non_iid_stats['A']['mean']:8.2f} | {non_iid_stats['A']['std']:8.2f} | {non_iid_stats['A']['min']:8.2f} | {non_iid_stats['A']['max']:8.2f} | {non_iid_stats['A']['n_samples']:,}
   B    | {non_iid_stats['B']['mean']:8.2f} | {non_iid_stats['B']['std']:8.2f} | {non_iid_stats['B']['min']:8.2f} | {non_iid_stats['B']['max']:8.2f} | {non_iid_stats['B']['n_samples']:,}
   C    | {non_iid_stats['C']['mean']:8.2f} | {non_iid_stats['C']['std']:8.2f} | {non_iid_stats['C']['min']:8.2f} | {non_iid_stats['C']['max']:8.2f} | {non_iid_stats['C']['n_samples']:,}

Visualizations Generated:
• 06_co2_histograms.png
• 06_co2_boxplot.png
• 06_co2_kde_comparison.png

================================================================================
OUTPUT FILES GENERATED
================================================================================
01_inspection_report.json          - Raw dataset inspection results
02_cleaning_log.json               - Detailed cleaning operations log
03_feature_engineering_info.json   - Feature extraction details
04_scaler.pkl                      - Fitted MinMaxScaler object
04_scaling_info.json               - Scaling parameters and statistics
05_X_school_A/B/C.npy              - Prepared input sequences
05_y_school_A/B/C.npy              - Prepared target values
05_preparation_info.json           - Data preparation details
06_non_iid_statistics.json         - Non-IID analysis statistics
06_co2_histograms.png              - Per-school CO2 distributions
06_co2_boxplot.png                 - Cross-school comparison boxplot
06_co2_kde_comparison.png          - Kernel density estimation overlay
07_scientific_interpretation.txt   - Scientific analysis of heterogeneity

================================================================================
READY FOR NEXT MISSION
================================================================================
All preprocessing steps completed. Data is ready for federated learning
experiments. Awaiting confirmation before proceeding to Mission 2.
================================================================================
"""

print(final_summary)

# Save final summary
with open(os.path.join(OUTPUT_PATH, "00_MISSION1_FINAL_SUMMARY.txt"), 'w') as f:
    f.write(final_summary)

print("\nFinal summary saved to: 00_MISSION1_FINAL_SUMMARY.txt")
print("\n" + "=" * 80)
print("MISSION 1 COMPLETE")
print("=" * 80)
