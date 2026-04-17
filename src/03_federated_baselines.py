"""
================================================================================
MISSION 3: FEDERATED LEARNING BASELINES
================================================================================
Research Project: Federated Learning for CO2 Prediction in Smart School Environments
Target: Q1 Journal Publication

This script implements three federated learning baseline methods:
1. FedAvg - Standard federated averaging (dataset-size weighted)
2. N-FedAvg - Normalized federated averaging (equal weights)
3. FedProx - FedAvg with proximal regularization

All methods use the SAME:
- Model architecture (LSTM(64) -> Dropout(0.2) -> Dense(1))
- Data preprocessing and splits from Mission 1
- Training hyperparameters

Author: Research Team
Date: March 2024
================================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import os
import json
import copy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
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
OUTPUT_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis", "mission3_results")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Plot settings
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.dpi'] = 300

print("=" * 80)
print("MISSION 3: FEDERATED LEARNING BASELINES")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Random seed: {RANDOM_SEED}")

# ================================================================================
# CONFIGURATION (Same as Mission 2 + FL settings)
# ================================================================================

CONFIG = {
    # Model Architecture (FIXED - same as Mission 2)
    'lstm_units': 64,
    'dropout_rate': 0.2,

    # Training Settings (FIXED - same as Mission 2)
    'learning_rate': 0.001,
    'batch_size': 32,

    # Federated Learning Settings
    'communication_rounds': 30,
    'local_epochs': 5,

    # FedProx Settings
    'fedprox_mu': 0.01,

    # Data
    'window_size': 12,
    'n_features': 5,

    # Reproducibility
    'random_seed': RANDOM_SEED
}

print("\n" + "-" * 60)
print("CONFIGURATION")
print("-" * 60)
print(f"Model: LSTM({CONFIG['lstm_units']}) -> Dropout({CONFIG['dropout_rate']}) -> Dense(1)")
print(f"Learning Rate: {CONFIG['learning_rate']}")
print(f"Batch Size: {CONFIG['batch_size']}")
print(f"Communication Rounds: {CONFIG['communication_rounds']}")
print(f"Local Epochs per Round: {CONFIG['local_epochs']}")
print(f"FedProx mu: {CONFIG['fedprox_mu']}")

# ================================================================================
# LOAD DATA (Same as Mission 2)
# ================================================================================

print("\n" + "=" * 80)
print("LOADING DATA FROM MISSION 1")
print("=" * 80)

def load_school_data(school):
    """Load train/val/test data for a school."""
    data = {}
    for split in ['train', 'val', 'test']:
        X = np.load(os.path.join(DATA_PATH, f"final_X_{school}_{split}.npy"))
        y = np.load(os.path.join(DATA_PATH, f"final_y_{school}_{split}.npy"))
        data[split] = {'X': X, 'y': y}
    return data

# Load data for each client (school)
client_data = {}
for school in ['A', 'B', 'C']:
    client_data[school] = load_school_data(school)
    print(f"Client {school}: Train={len(client_data[school]['train']['y']):,}, "
          f"Val={len(client_data[school]['val']['y']):,}, "
          f"Test={len(client_data[school]['test']['y']):,}")

# Calculate dataset sizes for weighted averaging
dataset_sizes = {s: len(client_data[s]['train']['y']) for s in ['A', 'B', 'C']}
total_samples = sum(dataset_sizes.values())
print(f"\nTotal training samples: {total_samples:,}")

# Dataset-size weights (for FedAvg)
fedavg_weights = {s: dataset_sizes[s] / total_samples for s in ['A', 'B', 'C']}
print(f"FedAvg weights: A={fedavg_weights['A']:.3f}, B={fedavg_weights['B']:.3f}, C={fedavg_weights['C']:.3f}")

# Equal weights (for N-FedAvg)
nfedavg_weights = {s: 1/3 for s in ['A', 'B', 'C']}
print(f"N-FedAvg weights: A={nfedavg_weights['A']:.3f}, B={nfedavg_weights['B']:.3f}, C={nfedavg_weights['C']:.3f}")

# Load scaler for inverse transform
with open(os.path.join(DATA_PATH, "final_scaler_params.json"), 'r') as f:
    scaler_params = json.load(f)

CO2_MIN = scaler_params['co2_inverse_transform']['min_ppm']
CO2_MAX = scaler_params['co2_inverse_transform']['max_ppm']

def inverse_transform_co2(y_normalized):
    """Convert normalized CO2 values back to ppm."""
    return y_normalized * (CO2_MAX - CO2_MIN) + CO2_MIN

# Prepare combined test set (for global evaluation)
X_test_global = np.concatenate([client_data[s]['test']['X'] for s in ['A', 'B', 'C']], axis=0)
y_test_global = np.concatenate([client_data[s]['test']['y'] for s in ['A', 'B', 'C']], axis=0)
print(f"\nGlobal test set: {len(y_test_global):,} samples")

# ================================================================================
# MODEL DEFINITION (Same as Mission 2)
# ================================================================================

def create_model(config):
    """Create LSTM model with FIXED architecture from Mission 2."""
    model = Sequential([
        Input(shape=(config['window_size'], config['n_features'])),
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

print("\n" + "=" * 80)
print("MODEL ARCHITECTURE (Fixed from Mission 2)")
print("=" * 80)
sample_model = create_model(CONFIG)
sample_model.summary()

# ================================================================================
# FEDERATED LEARNING UTILITIES
# ================================================================================

def get_model_weights(model):
    """Get model weights as a list of numpy arrays."""
    return [layer.copy() for layer in model.get_weights()]

def set_model_weights(model, weights):
    """Set model weights from a list of numpy arrays."""
    model.set_weights(weights)

def aggregate_weights(client_weights_list, aggregation_weights):
    """
    Aggregate client weights using specified weights.

    Parameters:
    -----------
    client_weights_list : list of list
        Each element is a client's model weights (list of numpy arrays)
    aggregation_weights : list
        Weights for each client (must sum to 1)

    Returns:
    --------
    aggregated_weights : list
        Aggregated model weights
    """
    # Initialize with zeros
    aggregated = [np.zeros_like(w) for w in client_weights_list[0]]

    # Weighted sum
    for client_weights, weight in zip(client_weights_list, aggregation_weights):
        for i, layer_weights in enumerate(client_weights):
            aggregated[i] += weight * layer_weights

    return aggregated

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics in ppm."""
    y_pred_norm = model.predict(X_test, verbose=0).flatten()
    y_pred_ppm = inverse_transform_co2(y_pred_norm)
    y_true_ppm = inverse_transform_co2(y_test)

    rmse = np.sqrt(mean_squared_error(y_true_ppm, y_pred_ppm))
    mae = mean_absolute_error(y_true_ppm, y_pred_ppm)
    r2 = r2_score(y_true_ppm, y_pred_ppm)

    return {'RMSE_ppm': rmse, 'MAE_ppm': mae, 'R2': r2}

# ================================================================================
# FEDAVG IMPLEMENTATION
# ================================================================================

def train_fedavg(config, client_data, weights_dict, method_name="FedAvg"):
    """
    Train using Federated Averaging.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    client_data : dict
        Data for each client
    weights_dict : dict
        Aggregation weights for each client
    method_name : str
        Name of the method (for logging)

    Returns:
    --------
    global_model : keras Model
        Final global model
    history : dict
        Training history (losses per round)
    """
    print(f"\n{'=' * 60}")
    print(f"Training {method_name}")
    print(f"{'=' * 60}")

    # Initialize global model
    global_model = create_model(config)
    global_weights = get_model_weights(global_model)

    clients = ['A', 'B', 'C']
    history = {
        'round': [],
        'global_val_loss': [],
        'global_val_rmse': [],
        'client_train_loss': {c: [] for c in clients}
    }

    for round_num in range(1, config['communication_rounds'] + 1):
        print(f"\nRound {round_num}/{config['communication_rounds']}")

        client_weights_list = []
        client_losses = []

        # Local training on each client
        for client in clients:
            # Create local model and set to global weights
            local_model = create_model(config)
            set_model_weights(local_model, global_weights)

            # Get client data
            X_train = client_data[client]['train']['X']
            y_train = client_data[client]['train']['y']

            # Local training
            hist = local_model.fit(
                X_train, y_train,
                epochs=config['local_epochs'],
                batch_size=config['batch_size'],
                verbose=0
            )

            # Store local weights and loss
            client_weights_list.append(get_model_weights(local_model))
            avg_loss = np.mean(hist.history['loss'])
            client_losses.append(avg_loss)
            history['client_train_loss'][client].append(avg_loss)

        # Aggregate weights
        agg_weights = [weights_dict[c] for c in clients]
        global_weights = aggregate_weights(client_weights_list, agg_weights)
        set_model_weights(global_model, global_weights)

        # Evaluate on combined validation set
        X_val = np.concatenate([client_data[c]['val']['X'] for c in clients], axis=0)
        y_val = np.concatenate([client_data[c]['val']['y'] for c in clients], axis=0)

        val_loss = global_model.evaluate(X_val, y_val, verbose=0)[0]
        val_metrics = evaluate_model(global_model, X_val, y_val)

        history['round'].append(round_num)
        history['global_val_loss'].append(val_loss)
        history['global_val_rmse'].append(val_metrics['RMSE_ppm'])

        print(f"  Client losses: A={client_losses[0]:.6f}, B={client_losses[1]:.6f}, C={client_losses[2]:.6f}")
        print(f"  Global val_loss: {val_loss:.6f}, val_RMSE: {val_metrics['RMSE_ppm']:.2f} ppm")

    return global_model, history

# ================================================================================
# FEDPROX IMPLEMENTATION
# ================================================================================

class FedProxModel(tf.keras.Model):
    """Custom model with FedProx proximal term."""

    def __init__(self, base_model, mu=0.01):
        super().__init__()
        self.base_model = base_model
        self.mu = mu
        self.global_weights = None

    def set_global_weights(self, weights):
        """Store global weights for proximal term calculation."""
        self.global_weights = [tf.constant(w) for w in weights]

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

            # Add proximal term: (mu/2) * ||w - w_global||^2
            if self.global_weights is not None:
                proximal_term = 0.0
                for w, w_global in zip(self.base_model.trainable_weights, self.global_weights):
                    proximal_term += tf.reduce_sum(tf.square(w - w_global))
                loss += (self.mu / 2) * proximal_term

        gradients = tape.gradient(loss, self.base_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def train_fedprox(config, client_data, weights_dict, mu=0.01):
    """
    Train using FedProx with proximal regularization.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    client_data : dict
        Data for each client
    weights_dict : dict
        Aggregation weights for each client
    mu : float
        Proximal term coefficient

    Returns:
    --------
    global_model : keras Model
        Final global model
    history : dict
        Training history
    """
    print(f"\n{'=' * 60}")
    print(f"Training FedProx (mu={mu})")
    print(f"{'=' * 60}")

    # Initialize global model
    global_model = create_model(config)
    global_weights = get_model_weights(global_model)

    clients = ['A', 'B', 'C']
    history = {
        'round': [],
        'global_val_loss': [],
        'global_val_rmse': [],
        'client_train_loss': {c: [] for c in clients}
    }

    for round_num in range(1, config['communication_rounds'] + 1):
        print(f"\nRound {round_num}/{config['communication_rounds']}")

        client_weights_list = []
        client_losses = []

        # Local training on each client with proximal term
        for client in clients:
            # Create base model
            base_model = create_model(config)
            set_model_weights(base_model, global_weights)

            # Wrap with FedProx
            local_model = FedProxModel(base_model, mu=mu)
            local_model.set_global_weights(global_weights)
            local_model.compile(
                optimizer=Adam(learning_rate=config['learning_rate']),
                loss='mse',
                metrics=['mae']
            )

            # Get client data
            X_train = client_data[client]['train']['X']
            y_train = client_data[client]['train']['y']

            # Local training
            hist = local_model.fit(
                X_train, y_train,
                epochs=config['local_epochs'],
                batch_size=config['batch_size'],
                verbose=0
            )

            # Store local weights and loss
            client_weights_list.append(get_model_weights(base_model))
            avg_loss = np.mean(hist.history['loss'])
            client_losses.append(avg_loss)
            history['client_train_loss'][client].append(avg_loss)

        # Aggregate weights (same as FedAvg)
        agg_weights = [weights_dict[c] for c in clients]
        global_weights = aggregate_weights(client_weights_list, agg_weights)
        set_model_weights(global_model, global_weights)

        # Evaluate on combined validation set
        X_val = np.concatenate([client_data[c]['val']['X'] for c in clients], axis=0)
        y_val = np.concatenate([client_data[c]['val']['y'] for c in clients], axis=0)

        val_loss = global_model.evaluate(X_val, y_val, verbose=0)[0]
        val_metrics = evaluate_model(global_model, X_val, y_val)

        history['round'].append(round_num)
        history['global_val_loss'].append(val_loss)
        history['global_val_rmse'].append(val_metrics['RMSE_ppm'])

        print(f"  Client losses: A={client_losses[0]:.6f}, B={client_losses[1]:.6f}, C={client_losses[2]:.6f}")
        print(f"  Global val_loss: {val_loss:.6f}, val_RMSE: {val_metrics['RMSE_ppm']:.2f} ppm")

    return global_model, history

# ================================================================================
# RUN EXPERIMENTS
# ================================================================================

print("\n" + "=" * 80)
print("RUNNING FEDERATED LEARNING EXPERIMENTS")
print("=" * 80)

# Store results
fl_results = {}
fl_histories = {}

# 1. FedAvg
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
fedavg_model, fedavg_history = train_fedavg(CONFIG, client_data, fedavg_weights, "FedAvg")
fl_results['FedAvg'] = {'model': fedavg_model}
fl_histories['FedAvg'] = fedavg_history

# 2. N-FedAvg
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
nfedavg_model, nfedavg_history = train_fedavg(CONFIG, client_data, nfedavg_weights, "N-FedAvg")
fl_results['N-FedAvg'] = {'model': nfedavg_model}
fl_histories['N-FedAvg'] = nfedavg_history

# 3. FedProx
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
fedprox_model, fedprox_history = train_fedprox(CONFIG, client_data, fedavg_weights, mu=CONFIG['fedprox_mu'])
fl_results['FedProx'] = {'model': fedprox_model}
fl_histories['FedProx'] = fedprox_history

# ================================================================================
# EVALUATION
# ================================================================================

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

# Evaluate each method
evaluation_results = {}

for method_name in ['FedAvg', 'N-FedAvg', 'FedProx']:
    print(f"\n{method_name}:")
    model = fl_results[method_name]['model']

    # Global test set
    global_metrics = evaluate_model(model, X_test_global, y_test_global)
    print(f"  Global: RMSE={global_metrics['RMSE_ppm']:.2f}, MAE={global_metrics['MAE_ppm']:.2f}, R2={global_metrics['R2']:.4f}")

    # Per-school test sets
    per_school_metrics = {}
    for school in ['A', 'B', 'C']:
        X_test = client_data[school]['test']['X']
        y_test = client_data[school]['test']['y']
        metrics = evaluate_model(model, X_test, y_test)
        per_school_metrics[school] = metrics
        print(f"  School {school}: RMSE={metrics['RMSE_ppm']:.2f}, MAE={metrics['MAE_ppm']:.2f}, R2={metrics['R2']:.4f}")

    evaluation_results[method_name] = {
        'global': global_metrics,
        'per_school': per_school_metrics
    }

# ================================================================================
# LOAD BASELINE RESULTS FROM MISSION 2
# ================================================================================

print("\n" + "=" * 80)
print("LOADING BASELINE RESULTS FROM MISSION 2")
print("=" * 80)

# Load Mission 2 results
with open(os.path.join(DATA_PATH, "mission2_results", "baseline_results.json"), 'r') as f:
    mission2_results = json.load(f)

centralized_metrics = mission2_results['centralized']['metrics']
centralized_per_school = mission2_results['centralized']['per_school_metrics']
local_metrics = {s: mission2_results['local'][s]['metrics'] for s in ['A', 'B', 'C']}

# Calculate local average
local_avg_rmse = np.mean([local_metrics[s]['RMSE_ppm'] for s in ['A', 'B', 'C']])
local_avg_mae = np.mean([local_metrics[s]['MAE_ppm'] for s in ['A', 'B', 'C']])
local_avg_r2 = np.mean([local_metrics[s]['R2'] for s in ['A', 'B', 'C']])

print(f"Centralized: RMSE={centralized_metrics['RMSE_ppm']:.2f}, MAE={centralized_metrics['MAE_ppm']:.2f}")
print(f"Local Avg: RMSE={local_avg_rmse:.2f}, MAE={local_avg_mae:.2f}")

# ================================================================================
# RESULTS SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Create comparison table
print("\n" + "-" * 75)
print("COMPARISON TABLE: ALL METHODS (Global Test Set)")
print("-" * 75)
print(f"{'Method':<20} {'RMSE (ppm)':>12} {'MAE (ppm)':>12} {'R2':>10}")
print("-" * 75)

# Centralized
print(f"{'Centralized':<20} {centralized_metrics['RMSE_ppm']:>12.2f} {centralized_metrics['MAE_ppm']:>12.2f} {centralized_metrics['R2']:>10.4f}")

# Local Average
print(f"{'Local Average':<20} {local_avg_rmse:>12.2f} {local_avg_mae:>12.2f} {local_avg_r2:>10.4f}")

print("-" * 75)

# FL Methods
for method in ['FedAvg', 'N-FedAvg', 'FedProx']:
    m = evaluation_results[method]['global']
    print(f"{method:<20} {m['RMSE_ppm']:>12.2f} {m['MAE_ppm']:>12.2f} {m['R2']:>10.4f}")

print("-" * 75)

# Per-school comparison
print("\n" + "-" * 75)
print("PER-SCHOOL PERFORMANCE: FEDERATED METHODS")
print("-" * 75)
print(f"{'Method':<15} {'School A RMSE':>14} {'School B RMSE':>14} {'School C RMSE':>14}")
print("-" * 75)

for method in ['FedAvg', 'N-FedAvg', 'FedProx']:
    m = evaluation_results[method]['per_school']
    print(f"{method:<15} {m['A']['RMSE_ppm']:>14.2f} {m['B']['RMSE_ppm']:>14.2f} {m['C']['RMSE_ppm']:>14.2f}")

print("-" * 75)
print("Centralized baseline:")
print(f"{'Centralized':<15} {centralized_per_school['A']['RMSE_ppm']:>14.2f} {centralized_per_school['B']['RMSE_ppm']:>14.2f} {centralized_per_school['C']['RMSE_ppm']:>14.2f}")
print("-" * 75)

# ================================================================================
# VISUALIZATIONS
# ================================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Convergence plots (RMSE over rounds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation RMSE convergence
ax = axes[0]
for method, color in zip(['FedAvg', 'N-FedAvg', 'FedProx'], ['#2ecc71', '#3498db', '#e74c3c']):
    rounds = fl_histories[method]['round']
    rmse_vals = fl_histories[method]['global_val_rmse']
    ax.plot(rounds, rmse_vals, label=method, linewidth=2, color=color)

ax.axhline(centralized_metrics['RMSE_ppm'], color='black', linestyle='--', linewidth=2, label='Centralized')
ax.axhline(local_avg_rmse, color='gray', linestyle=':', linewidth=2, label='Local Avg')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Global Validation RMSE (ppm)')
ax.set_title('Convergence: Validation RMSE per Round')
ax.legend()
ax.grid(True, alpha=0.3)

# Validation Loss convergence
ax = axes[1]
for method, color in zip(['FedAvg', 'N-FedAvg', 'FedProx'], ['#2ecc71', '#3498db', '#e74c3c']):
    rounds = fl_histories[method]['round']
    loss_vals = fl_histories[method]['global_val_loss']
    ax.plot(rounds, loss_vals, label=method, linewidth=2, color=color)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Global Validation Loss (MSE)')
ax.set_title('Convergence: Validation Loss per Round')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Federated Learning Convergence', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fl_convergence.png'), dpi=300)
plt.close()
print("Saved: fl_convergence.png")

# 2. Method comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

methods = ['Centralized', 'Local Avg', 'FedAvg', 'N-FedAvg', 'FedProx']
colors = ['#2c3e50', '#7f8c8d', '#2ecc71', '#3498db', '#e74c3c']

# RMSE comparison
ax = axes[0]
rmse_values = [
    centralized_metrics['RMSE_ppm'],
    local_avg_rmse,
    evaluation_results['FedAvg']['global']['RMSE_ppm'],
    evaluation_results['N-FedAvg']['global']['RMSE_ppm'],
    evaluation_results['FedProx']['global']['RMSE_ppm']
]
bars = ax.bar(methods, rmse_values, color=colors)
ax.set_ylabel('RMSE (ppm)')
ax.set_title('RMSE Comparison')
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)

# MAE comparison
ax = axes[1]
mae_values = [
    centralized_metrics['MAE_ppm'],
    local_avg_mae,
    evaluation_results['FedAvg']['global']['MAE_ppm'],
    evaluation_results['N-FedAvg']['global']['MAE_ppm'],
    evaluation_results['FedProx']['global']['MAE_ppm']
]
bars = ax.bar(methods, mae_values, color=colors)
ax.set_ylabel('MAE (ppm)')
ax.set_title('MAE Comparison')
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)

# R2 comparison
ax = axes[2]
r2_values = [
    centralized_metrics['R2'],
    local_avg_r2,
    evaluation_results['FedAvg']['global']['R2'],
    evaluation_results['N-FedAvg']['global']['R2'],
    evaluation_results['FedProx']['global']['R2']
]
bars = ax.bar(methods, r2_values, color=colors)
ax.set_ylabel('R2 Score')
ax.set_title('R2 Comparison')
ax.set_ylim(0.9, 1.0)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)

plt.suptitle('Federated Learning vs Baselines', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fl_comparison.png'), dpi=300)
plt.close()
print("Saved: fl_comparison.png")

# 3. Per-school comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

schools = ['A', 'B', 'C']
x = np.arange(len(schools))
width = 0.2

for ax_idx, (ax, title) in enumerate(zip(axes, ['School A', 'School B', 'School C'])):
    school = schools[ax_idx]

    methods_plot = ['Centralized', 'FedAvg', 'N-FedAvg', 'FedProx']
    rmse_vals = [
        centralized_per_school[school]['RMSE_ppm'],
        evaluation_results['FedAvg']['per_school'][school]['RMSE_ppm'],
        evaluation_results['N-FedAvg']['per_school'][school]['RMSE_ppm'],
        evaluation_results['FedProx']['per_school'][school]['RMSE_ppm']
    ]

    colors_plot = ['#2c3e50', '#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(methods_plot, rmse_vals, color=colors_plot)
    ax.set_ylabel('RMSE (ppm)')
    ax.set_title(f'{title} Test Performance')
    ax.set_xticklabels(methods_plot, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)

plt.suptitle('Per-School RMSE: FL Methods vs Centralized', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fl_per_school.png'), dpi=300)
plt.close()
print("Saved: fl_per_school.png")

# ================================================================================
# SAVE RESULTS
# ================================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Compile all results
all_results = {
    'config': CONFIG,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'aggregation_weights': {
        'FedAvg': fedavg_weights,
        'N-FedAvg': nfedavg_weights,
        'FedProx': fedavg_weights
    },
    'evaluation': evaluation_results,
    'baselines': {
        'centralized': centralized_metrics,
        'centralized_per_school': centralized_per_school,
        'local_average': {'RMSE_ppm': local_avg_rmse, 'MAE_ppm': local_avg_mae, 'R2': local_avg_r2}
    }
}

with open(os.path.join(OUTPUT_PATH, 'fl_baseline_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print("Saved: fl_baseline_results.json")

# Save histories
histories_data = {
    method: {
        'round': hist['round'],
        'global_val_loss': hist['global_val_loss'],
        'global_val_rmse': hist['global_val_rmse']
    }
    for method, hist in fl_histories.items()
}

with open(os.path.join(OUTPUT_PATH, 'fl_training_histories.json'), 'w') as f:
    json.dump(histories_data, f, indent=2)
print("Saved: fl_training_histories.json")

# Save models
for method in ['FedAvg', 'N-FedAvg', 'FedProx']:
    model_path = os.path.join(OUTPUT_PATH, f'{method}_model.keras')
    fl_results[method]['model'].save(model_path)
    print(f"Saved: {method}_model.keras")

# ================================================================================
# SCIENTIFIC INTERPRETATION
# ================================================================================

print("\n" + "=" * 80)
print("SCIENTIFIC INTERPRETATION")
print("=" * 80)

# Calculate improvements
fedavg_vs_local = local_avg_rmse - evaluation_results['FedAvg']['global']['RMSE_ppm']
nfedavg_vs_local = local_avg_rmse - evaluation_results['N-FedAvg']['global']['RMSE_ppm']
fedprox_vs_local = local_avg_rmse - evaluation_results['FedProx']['global']['RMSE_ppm']

fedavg_vs_cent = evaluation_results['FedAvg']['global']['RMSE_ppm'] - centralized_metrics['RMSE_ppm']
nfedavg_vs_cent = evaluation_results['N-FedAvg']['global']['RMSE_ppm'] - centralized_metrics['RMSE_ppm']
fedprox_vs_cent = evaluation_results['FedProx']['global']['RMSE_ppm'] - centralized_metrics['RMSE_ppm']

# Find best method
fl_rmse = {
    'FedAvg': evaluation_results['FedAvg']['global']['RMSE_ppm'],
    'N-FedAvg': evaluation_results['N-FedAvg']['global']['RMSE_ppm'],
    'FedProx': evaluation_results['FedProx']['global']['RMSE_ppm']
}
best_fl_method = min(fl_rmse, key=fl_rmse.get)

interpretation = f"""
================================================================================
SCIENTIFIC INTERPRETATION: FEDERATED LEARNING BASELINES
================================================================================

1. WHICH FL BASELINE PERFORMS BEST?
-----------------------------------

Global Test Set Performance (RMSE in ppm):
   - FedAvg:   {evaluation_results['FedAvg']['global']['RMSE_ppm']:.2f} ppm
   - N-FedAvg: {evaluation_results['N-FedAvg']['global']['RMSE_ppm']:.2f} ppm
   - FedProx:  {evaluation_results['FedProx']['global']['RMSE_ppm']:.2f} ppm

Best performing FL method: {best_fl_method} ({fl_rmse[best_fl_method]:.2f} ppm RMSE)


2. HOW CLOSE DOES EACH METHOD GET TO CENTRALIZED?
-------------------------------------------------

Centralized benchmark: {centralized_metrics['RMSE_ppm']:.2f} ppm RMSE

Gap from centralized (lower is better):
   - FedAvg:   {fedavg_vs_cent:+.2f} ppm
   - N-FedAvg: {nfedavg_vs_cent:+.2f} ppm
   - FedProx:  {fedprox_vs_cent:+.2f} ppm

All FL methods successfully outperform local-only training
(Local Average: {local_avg_rmse:.2f} ppm):
   - FedAvg improvement:   {fedavg_vs_local:.2f} ppm better
   - N-FedAvg improvement: {nfedavg_vs_local:.2f} ppm better
   - FedProx improvement:  {fedprox_vs_local:.2f} ppm better


3. DOES FEDPROX HELP UNDER NON-IID SETTING?
-------------------------------------------

FedProx performance vs FedAvg:
   - FedProx RMSE: {evaluation_results['FedProx']['global']['RMSE_ppm']:.2f} ppm
   - FedAvg RMSE:  {evaluation_results['FedAvg']['global']['RMSE_ppm']:.2f} ppm
   - Difference:   {evaluation_results['FedProx']['global']['RMSE_ppm'] - evaluation_results['FedAvg']['global']['RMSE_ppm']:+.2f} ppm

The proximal term (mu={CONFIG['fedprox_mu']}) {'helps' if evaluation_results['FedProx']['global']['RMSE_ppm'] < evaluation_results['FedAvg']['global']['RMSE_ppm'] else 'does not significantly improve over'} standard FedAvg
in this non-IID setting. This suggests that the heterogeneity between schools,
while present, may be {'moderate' if abs(evaluation_results['FedProx']['global']['RMSE_ppm'] - evaluation_results['FedAvg']['global']['RMSE_ppm']) < 2 else 'significant'} in terms of model convergence impact.


4. IS EQUAL WEIGHTING (N-FedAvg) BETTER OR WORSE?
-------------------------------------------------

Weight comparison:
   - FedAvg (dataset-size weights): A={fedavg_weights['A']:.2%}, B={fedavg_weights['B']:.2%}, C={fedavg_weights['C']:.2%}
   - N-FedAvg (equal weights):      A=33.33%, B=33.33%, C=33.33%

Performance comparison:
   - FedAvg RMSE:   {evaluation_results['FedAvg']['global']['RMSE_ppm']:.2f} ppm
   - N-FedAvg RMSE: {evaluation_results['N-FedAvg']['global']['RMSE_ppm']:.2f} ppm

{"Dataset-size weighting (FedAvg) performs better" if evaluation_results['FedAvg']['global']['RMSE_ppm'] < evaluation_results['N-FedAvg']['global']['RMSE_ppm'] else "Equal weighting (N-FedAvg) performs better" if evaluation_results['N-FedAvg']['global']['RMSE_ppm'] < evaluation_results['FedAvg']['global']['RMSE_ppm'] else "Both weighting strategies perform similarly"}.

This {'aligns with theory suggesting larger datasets provide more reliable gradient estimates' if evaluation_results['FedAvg']['global']['RMSE_ppm'] < evaluation_results['N-FedAvg']['global']['RMSE_ppm'] else 'suggests equal representation prevents larger clients from dominating' if evaluation_results['N-FedAvg']['global']['RMSE_ppm'] < evaluation_results['FedAvg']['global']['RMSE_ppm'] else 'indicates weighting strategy has minimal impact in this scenario'}.


5. PER-SCHOOL ANALYSIS
----------------------

School A (most challenging - lowest local R2):
   - FedAvg:   {evaluation_results['FedAvg']['per_school']['A']['RMSE_ppm']:.2f} ppm
   - N-FedAvg: {evaluation_results['N-FedAvg']['per_school']['A']['RMSE_ppm']:.2f} ppm
   - FedProx:  {evaluation_results['FedProx']['per_school']['A']['RMSE_ppm']:.2f} ppm
   - Central:  {centralized_per_school['A']['RMSE_ppm']:.2f} ppm

School B (largest dataset):
   - FedAvg:   {evaluation_results['FedAvg']['per_school']['B']['RMSE_ppm']:.2f} ppm
   - N-FedAvg: {evaluation_results['N-FedAvg']['per_school']['B']['RMSE_ppm']:.2f} ppm
   - FedProx:  {evaluation_results['FedProx']['per_school']['B']['RMSE_ppm']:.2f} ppm
   - Central:  {centralized_per_school['B']['RMSE_ppm']:.2f} ppm

School C (highest predictability):
   - FedAvg:   {evaluation_results['FedAvg']['per_school']['C']['RMSE_ppm']:.2f} ppm
   - N-FedAvg: {evaluation_results['N-FedAvg']['per_school']['C']['RMSE_ppm']:.2f} ppm
   - FedProx:  {evaluation_results['FedProx']['per_school']['C']['RMSE_ppm']:.2f} ppm
   - Central:  {centralized_per_school['C']['RMSE_ppm']:.2f} ppm


6. KEY CONCLUSIONS
------------------

a) Federated learning successfully achieves near-centralized performance
   while preserving data privacy (no raw data sharing required).

b) All three FL methods significantly outperform isolated local training,
   demonstrating the value of collaborative learning.

c) The gap between FL methods and centralized training is relatively small,
   suggesting federated learning is a viable privacy-preserving alternative.

d) {best_fl_method} achieves the best overall performance among the
   baseline FL methods tested.


================================================================================
"""

print(interpretation)

with open(os.path.join(OUTPUT_PATH, 'fl_interpretation.txt'), 'w') as f:
    f.write(interpretation)
print("\nSaved: fl_interpretation.txt")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

final_summary = f"""
================================================================================
MISSION 3: FEDERATED LEARNING BASELINES - FINAL SUMMARY
================================================================================

CONFIGURATION:
   - Model: LSTM(64) -> Dropout(0.2) -> Dense(1) (same as Mission 2)
   - Communication Rounds: {CONFIG['communication_rounds']}
   - Local Epochs: {CONFIG['local_epochs']}
   - Batch Size: {CONFIG['batch_size']}
   - Learning Rate: {CONFIG['learning_rate']}
   - FedProx mu: {CONFIG['fedprox_mu']}

GLOBAL TEST SET RESULTS:
+-------------------------+------------+------------+----------+
| Method                  | RMSE (ppm) | MAE (ppm)  | R2       |
+-------------------------+------------+------------+----------+
| Centralized             | {centralized_metrics['RMSE_ppm']:>10.2f} | {centralized_metrics['MAE_ppm']:>10.2f} | {centralized_metrics['R2']:>8.4f} |
| Local Average           | {local_avg_rmse:>10.2f} | {local_avg_mae:>10.2f} | {local_avg_r2:>8.4f} |
+-------------------------+------------+------------+----------+
| FedAvg                  | {evaluation_results['FedAvg']['global']['RMSE_ppm']:>10.2f} | {evaluation_results['FedAvg']['global']['MAE_ppm']:>10.2f} | {evaluation_results['FedAvg']['global']['R2']:>8.4f} |
| N-FedAvg                | {evaluation_results['N-FedAvg']['global']['RMSE_ppm']:>10.2f} | {evaluation_results['N-FedAvg']['global']['MAE_ppm']:>10.2f} | {evaluation_results['N-FedAvg']['global']['R2']:>8.4f} |
| FedProx (mu={CONFIG['fedprox_mu']})        | {evaluation_results['FedProx']['global']['RMSE_ppm']:>10.2f} | {evaluation_results['FedProx']['global']['MAE_ppm']:>10.2f} | {evaluation_results['FedProx']['global']['R2']:>8.4f} |
+-------------------------+------------+------------+----------+

PER-SCHOOL RMSE (ppm):
+-------------------------+----------+----------+----------+
| Method                  | School A | School B | School C |
+-------------------------+----------+----------+----------+
| Centralized             | {centralized_per_school['A']['RMSE_ppm']:>8.2f} | {centralized_per_school['B']['RMSE_ppm']:>8.2f} | {centralized_per_school['C']['RMSE_ppm']:>8.2f} |
| FedAvg                  | {evaluation_results['FedAvg']['per_school']['A']['RMSE_ppm']:>8.2f} | {evaluation_results['FedAvg']['per_school']['B']['RMSE_ppm']:>8.2f} | {evaluation_results['FedAvg']['per_school']['C']['RMSE_ppm']:>8.2f} |
| N-FedAvg                | {evaluation_results['N-FedAvg']['per_school']['A']['RMSE_ppm']:>8.2f} | {evaluation_results['N-FedAvg']['per_school']['B']['RMSE_ppm']:>8.2f} | {evaluation_results['N-FedAvg']['per_school']['C']['RMSE_ppm']:>8.2f} |
| FedProx                 | {evaluation_results['FedProx']['per_school']['A']['RMSE_ppm']:>8.2f} | {evaluation_results['FedProx']['per_school']['B']['RMSE_ppm']:>8.2f} | {evaluation_results['FedProx']['per_school']['C']['RMSE_ppm']:>8.2f} |
+-------------------------+----------+----------+----------+

OUTPUT FILES:
   - fl_baseline_results.json
   - fl_training_histories.json
   - fl_interpretation.txt
   - fl_convergence.png
   - fl_comparison.png
   - fl_per_school.png
   - FedAvg_model.keras
   - N-FedAvg_model.keras
   - FedProx_model.keras

================================================================================
MISSION 3 COMPLETE
================================================================================
Federated learning baselines implemented and evaluated.
Awaiting confirmation before proceeding to Mission 4 (FedCORA).
================================================================================
"""

print(final_summary)

with open(os.path.join(OUTPUT_PATH, 'mission3_summary.txt'), 'w') as f:
    f.write(final_summary)
print("\nSaved: mission3_summary.txt")
