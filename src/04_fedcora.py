"""
================================================================================
MISSION 4: FedCORA - FEDERATED CONTEXT-ORIENTED RELIABLE AGGREGATION
================================================================================
Main contribution for Q1 Journal Paper

FedCORA is designed to improve the trade-off between robustness and efficiency
under non-IID school environments through:

1. Adaptive Local Stabilization - selective proximal regularization
2. Context-Aware Fuzzy Aggregation - intelligent weight assignment
3. Reduced Computation - 3 local epochs vs 5

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
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import functools
print = functools.partial(print, flush=True)

# ================================================================================
# PATHS
# ================================================================================
BASE_PATH = r"C:\Users\info\Documents\my-project\Federated Learning"
DATA_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis")
OUTPUT_PATH = os.path.join(BASE_PATH, "CO2_Preprocessing_Analysis", "mission4_fedcora")
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 80)
print("MISSION 4: FedCORA IMPLEMENTATION")
print("Federated Context-Oriented Reliable Aggregation")
print("=" * 80)
print(f"TensorFlow: {tf.__version__}")

# ================================================================================
# CONFIGURATION
# ================================================================================

CONFIG = {
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'window_size': 12,
    'n_features': 5,
    'communication_rounds': 30,
    'local_epochs': 3,          # Reduced from 5 for efficiency
    'mu': 0.001,                # Proximal term (only for unstable clients)
    'seeds': [42, 123, 456, 789, 2024],
}

# Fixed scenario values (domain knowledge)
SCENARIO = {
    'A': {'student_density': 0.28, 'ventilation_adequacy': 0.82},
    'B': {'student_density': 0.45, 'ventilation_adequacy': 0.56},
    'C': {'student_density': 0.62, 'ventilation_adequacy': 0.33},
}

print(f"\nConfig: {len(CONFIG['seeds'])} seeds, {CONFIG['communication_rounds']} rounds")
print(f"Local epochs: {CONFIG['local_epochs']} (efficiency optimized)")
print(f"Proximal mu: {CONFIG['mu']} (for unstable clients only)")

# ================================================================================
# LOAD DATA
# ================================================================================

print("\n" + "=" * 80)
print("[1] LOADING DATA")
print("=" * 80)

def load_data():
    client_data = {}
    for school in ['A', 'B', 'C']:
        client_data[school] = {}
        for split in ['train', 'val', 'test']:
            X = np.load(os.path.join(DATA_PATH, f"final_X_{school}_{split}.npy"))
            y = np.load(os.path.join(DATA_PATH, f"final_y_{school}_{split}.npy"))
            client_data[school][split] = {'X': X, 'y': y}
    return client_data

client_data = load_data()

with open(os.path.join(DATA_PATH, "final_scaler_params.json"), 'r') as f:
    scaler_params = json.load(f)

CO2_MIN = scaler_params['co2_inverse_transform']['min_ppm']
CO2_MAX = scaler_params['co2_inverse_transform']['max_ppm']

def inverse_transform_co2(y_normalized):
    return y_normalized * (CO2_MAX - CO2_MIN) + CO2_MIN

# Combined datasets
X_train_cent = np.concatenate([client_data[s]['train']['X'] for s in ['A', 'B', 'C']], axis=0)
y_train_cent = np.concatenate([client_data[s]['train']['y'] for s in ['A', 'B', 'C']], axis=0)
X_val_cent = np.concatenate([client_data[s]['val']['X'] for s in ['A', 'B', 'C']], axis=0)
y_val_cent = np.concatenate([client_data[s]['val']['y'] for s in ['A', 'B', 'C']], axis=0)
X_test_global = np.concatenate([client_data[s]['test']['X'] for s in ['A', 'B', 'C']], axis=0)
y_test_global = np.concatenate([client_data[s]['test']['y'] for s in ['A', 'B', 'C']], axis=0)

print(f"Data loaded: train={len(y_train_cent)}, val={len(y_val_cent)}, test={len(y_test_global)}")

for school in ['A', 'B', 'C']:
    n_train = len(client_data[school]['train']['y'])
    print(f"  School {school}: {n_train} train samples")

# ================================================================================
# ENVIRONMENTAL RELIABILITY INDEX (ERI) CALCULATION
# ================================================================================

print("\n" + "=" * 80)
print("[2] COMPUTING ENVIRONMENTAL RELIABILITY INDEX (ERI)")
print("=" * 80)

def compute_eri(y_data, X_data):
    """
    Compute Environmental Reliability Index from real data.
    ERI = 0.4*(1 - normalized_variance) + 0.3*(1 - missing_ratio) + 0.3*(1 - anomaly_ratio)

    Note: Since data is already preprocessed, we estimate these from available metrics.
    """
    # 1. Normalized variance (higher variance = less reliable)
    y_ppm = inverse_transform_co2(y_data)
    variance = np.var(y_ppm)
    # Normalize using expected range (0-2000 ppm typical variance range)
    max_expected_variance = 200000  # ~450 ppm std
    normalized_variance = min(variance / max_expected_variance, 1.0)

    # 2. Missing ratio - assume 0 since data is preprocessed
    missing_ratio = 0.0

    # 3. Anomaly ratio - estimate from outliers (values beyond 2 std)
    mean_val = np.mean(y_ppm)
    std_val = np.std(y_ppm)
    outliers = np.sum((y_ppm < mean_val - 2*std_val) | (y_ppm > mean_val + 2*std_val))
    anomaly_ratio = outliers / len(y_ppm)

    # Compute ERI
    eri = 0.4 * (1 - normalized_variance) + 0.3 * (1 - missing_ratio) + 0.3 * (1 - anomaly_ratio)

    return {
        'eri': eri,
        'normalized_variance': normalized_variance,
        'missing_ratio': missing_ratio,
        'anomaly_ratio': anomaly_ratio
    }

# Compute ERI for each school
ERI_DATA = {}
for school in ['A', 'B', 'C']:
    eri_info = compute_eri(
        client_data[school]['train']['y'],
        client_data[school]['train']['X']
    )
    ERI_DATA[school] = eri_info
    print(f"School {school}: ERI = {eri_info['eri']:.4f}")
    print(f"  - Normalized variance: {eri_info['normalized_variance']:.4f}")
    print(f"  - Missing ratio: {eri_info['missing_ratio']:.4f}")
    print(f"  - Anomaly ratio: {eri_info['anomaly_ratio']:.4f}")

# ================================================================================
# FUZZY SYSTEM FOR AGGREGATION WEIGHTS
# ================================================================================

print("\n" + "=" * 80)
print("[3] FUZZY SYSTEM DESIGN")
print("=" * 80)

class FuzzyAggregator:
    """
    Simple and interpretable fuzzy system for computing aggregation weights.

    Inputs (4):
        1. Student Density: LOW, MEDIUM, HIGH
        2. Ventilation Adequacy: POOR, MODERATE, GOOD
        3. ERI: LOW, MEDIUM, HIGH
        4. Local RMSE: LOW, MEDIUM, HIGH

    Output:
        Aggregation Weight: LOW, MEDIUM, HIGH
    """

    def __init__(self):
        # Define membership function parameters (triangular)
        # Format: (a, b, c) where a=start, b=peak, c=end

        # Student Density (0-1): LOW=[0,0.3], MEDIUM=[0.2,0.6], HIGH=[0.5,1]
        self.density_mf = {
            'LOW': (0.0, 0.0, 0.35),
            'MEDIUM': (0.25, 0.45, 0.65),
            'HIGH': (0.55, 1.0, 1.0)
        }

        # Ventilation Adequacy (0-1): POOR=[0,0.4], MODERATE=[0.3,0.7], GOOD=[0.6,1]
        self.ventilation_mf = {
            'POOR': (0.0, 0.0, 0.45),
            'MODERATE': (0.35, 0.55, 0.75),
            'GOOD': (0.65, 1.0, 1.0)
        }

        # ERI (0-1): LOW=[0,0.5], MEDIUM=[0.4,0.7], HIGH=[0.6,1]
        self.eri_mf = {
            'LOW': (0.0, 0.0, 0.55),
            'MEDIUM': (0.45, 0.65, 0.85),
            'HIGH': (0.75, 1.0, 1.0)
        }

        # RMSE (normalized 0-1): LOW=[0,0.3], MEDIUM=[0.2,0.6], HIGH=[0.5,1]
        self.rmse_mf = {
            'LOW': (0.0, 0.0, 0.35),
            'MEDIUM': (0.25, 0.5, 0.75),
            'HIGH': (0.65, 1.0, 1.0)
        }

        # Output weight levels
        self.weight_values = {'LOW': 0.15, 'MEDIUM': 0.35, 'HIGH': 0.55}

    def triangular_mf(self, x, params):
        """Triangular membership function."""
        a, b, c = params
        if x <= a:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b < x < c:
            return (c - x) / (c - b) if c != b else 1.0
        else:
            return 0.0

    def fuzzify(self, value, mf_dict):
        """Fuzzify a value into membership degrees."""
        return {level: self.triangular_mf(value, params)
                for level, params in mf_dict.items()}

    def apply_rules(self, density_fuzzy, ventilation_fuzzy, eri_fuzzy, rmse_fuzzy):
        """
        Apply fuzzy rules - simplified rule base for interpretability.

        Key principles:
        - High ERI + Low RMSE -> HIGH weight
        - Low ERI -> LOWER weight
        - High RMSE -> LOWER weight
        - Good ventilation + Low density -> higher weight
        - Poor ventilation + High density -> lower weight
        """
        output_weights = {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 0.0}

        # Rule 1: HIGH ERI AND LOW RMSE -> HIGH weight
        r1 = min(eri_fuzzy['HIGH'], rmse_fuzzy['LOW'])
        output_weights['HIGH'] = max(output_weights['HIGH'], r1)

        # Rule 2: HIGH ERI AND MEDIUM RMSE -> MEDIUM-HIGH weight
        r2 = min(eri_fuzzy['HIGH'], rmse_fuzzy['MEDIUM'])
        output_weights['MEDIUM'] = max(output_weights['MEDIUM'], r2 * 0.7)
        output_weights['HIGH'] = max(output_weights['HIGH'], r2 * 0.3)

        # Rule 3: MEDIUM ERI AND LOW RMSE -> MEDIUM-HIGH weight
        r3 = min(eri_fuzzy['MEDIUM'], rmse_fuzzy['LOW'])
        output_weights['MEDIUM'] = max(output_weights['MEDIUM'], r3 * 0.4)
        output_weights['HIGH'] = max(output_weights['HIGH'], r3 * 0.6)

        # Rule 4: MEDIUM ERI AND MEDIUM RMSE -> MEDIUM weight
        r4 = min(eri_fuzzy['MEDIUM'], rmse_fuzzy['MEDIUM'])
        output_weights['MEDIUM'] = max(output_weights['MEDIUM'], r4)

        # Rule 5: LOW ERI -> LOW-MEDIUM weight (regardless of RMSE)
        r5 = eri_fuzzy['LOW']
        output_weights['LOW'] = max(output_weights['LOW'], r5 * 0.6)
        output_weights['MEDIUM'] = max(output_weights['MEDIUM'], r5 * 0.4)

        # Rule 6: HIGH RMSE -> LOW weight
        r6 = rmse_fuzzy['HIGH']
        output_weights['LOW'] = max(output_weights['LOW'], r6 * 0.7)
        output_weights['MEDIUM'] = max(output_weights['MEDIUM'], r6 * 0.3)

        # Rule 7: GOOD ventilation AND LOW density -> boost weight
        r7 = min(ventilation_fuzzy['GOOD'], density_fuzzy['LOW'])
        output_weights['HIGH'] = max(output_weights['HIGH'], r7 * 0.4)

        # Rule 8: POOR ventilation AND HIGH density -> reduce weight
        r8 = min(ventilation_fuzzy['POOR'], density_fuzzy['HIGH'])
        output_weights['LOW'] = max(output_weights['LOW'], r8 * 0.5)

        # Rule 9: GOOD ventilation AND MEDIUM density AND HIGH ERI -> HIGH
        r9 = min(ventilation_fuzzy['GOOD'], density_fuzzy['MEDIUM'], eri_fuzzy['HIGH'])
        output_weights['HIGH'] = max(output_weights['HIGH'], r9 * 0.5)

        # Rule 10: MODERATE ventilation AND MEDIUM ERI -> MEDIUM
        r10 = min(ventilation_fuzzy['MODERATE'], eri_fuzzy['MEDIUM'])
        output_weights['MEDIUM'] = max(output_weights['MEDIUM'], r10 * 0.6)

        return output_weights

    def defuzzify(self, output_weights):
        """Center of gravity defuzzification."""
        total_activation = sum(output_weights.values())
        if total_activation == 0:
            return 0.33  # Default equal weight

        weighted_sum = sum(self.weight_values[level] * activation
                          for level, activation in output_weights.items())
        return weighted_sum / total_activation

    def compute_weight(self, density, ventilation, eri, rmse_normalized):
        """Compute aggregation weight for a client."""
        # Fuzzify inputs
        density_fuzzy = self.fuzzify(density, self.density_mf)
        ventilation_fuzzy = self.fuzzify(ventilation, self.ventilation_mf)
        eri_fuzzy = self.fuzzify(eri, self.eri_mf)
        rmse_fuzzy = self.fuzzify(rmse_normalized, self.rmse_mf)

        # Apply rules
        output_weights = self.apply_rules(density_fuzzy, ventilation_fuzzy,
                                          eri_fuzzy, rmse_fuzzy)

        # Defuzzify
        raw_weight = self.defuzzify(output_weights)

        return {
            'raw_weight': raw_weight,
            'density_fuzzy': density_fuzzy,
            'ventilation_fuzzy': ventilation_fuzzy,
            'eri_fuzzy': eri_fuzzy,
            'rmse_fuzzy': rmse_fuzzy,
            'output_weights': output_weights
        }

fuzzy_aggregator = FuzzyAggregator()
print("Fuzzy system initialized with 10 interpretable rules")

# ================================================================================
# STABILITY CLASSIFIER
# ================================================================================

print("\n" + "=" * 80)
print("[4] STABILITY CLASSIFICATION")
print("=" * 80)

def classify_stability(eri, rmse_ppm, eri_threshold=0.7, rmse_threshold=80):
    """
    Classify client as STABLE or UNSTABLE.

    STABLE: high ERI AND low RMSE
    UNSTABLE: low ERI OR high RMSE

    Args:
        eri: Environmental Reliability Index
        rmse_ppm: Local validation RMSE in ppm
        eri_threshold: Threshold for "high" ERI
        rmse_threshold: Threshold for "low" RMSE (in ppm)

    Returns:
        dict with stability classification
    """
    is_high_eri = eri >= eri_threshold
    is_low_rmse = rmse_ppm <= rmse_threshold

    is_stable = is_high_eri and is_low_rmse

    return {
        'stable': is_stable,
        'high_eri': is_high_eri,
        'low_rmse': is_low_rmse,
        'use_proximal': not is_stable
    }

print("Stability classifier initialized")
print(f"  ERI threshold: >= 0.7 for stability")
print(f"  RMSE threshold: <= 80 ppm for stability")

# ================================================================================
# MODEL UTILITIES
# ================================================================================

def create_model():
    model = Sequential([
        Input(shape=(CONFIG['window_size'], CONFIG['n_features'])),
        LSTM(CONFIG['lstm_units'], return_sequences=False),
        Dropout(CONFIG['dropout_rate']),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']),
                  loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X, verbose=0).flatten()
    y_pred_ppm = inverse_transform_co2(y_pred)
    y_true_ppm = inverse_transform_co2(y)
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true_ppm, y_pred_ppm)),
        'MAE': mean_absolute_error(y_true_ppm, y_pred_ppm),
        'R2': r2_score(y_true_ppm, y_pred_ppm)
    }

def get_weights(model):
    return [w.copy() for w in model.get_weights()]

def set_weights(model, weights):
    model.set_weights(weights)

# ================================================================================
# ADAPTIVE LOCAL TRAINER (STABLE vs UNSTABLE)
# ================================================================================

class AdaptiveLocalTrainer:
    """
    Adaptive local training for FedCORA.
    - STABLE clients: standard training (model.fit)
    - UNSTABLE clients: proximal regularization
    """

    def __init__(self, model, mu, learning_rate, use_proximal=False):
        self.model = model
        self.mu = mu
        self.use_proximal = use_proximal
        self.optimizer = Adam(learning_rate=learning_rate)
        self.global_weights = None

    def set_global_weights(self, weights):
        if self.use_proximal:
            self.global_weights = [tf.Variable(w, trainable=False) for w in weights]

    @tf.function
    def proximal_train_step(self, X_batch, y_batch):
        """Training step with proximal regularization (for unstable clients)."""
        with tf.GradientTape() as tape:
            y_pred = self.model(X_batch, training=True)
            mse_loss = tf.reduce_mean(tf.square(y_batch - y_pred))

            # Proximal term
            prox_term = tf.constant(0.0)
            for w, w_g in zip(self.model.trainable_weights, self.global_weights):
                prox_term = prox_term + tf.reduce_sum(tf.square(w - w_g))

            total_loss = mse_loss + (self.mu / 2.0) * prox_term

        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return mse_loss

    def train(self, X, y, epochs, batch_size):
        """Train the local model."""
        if self.use_proximal:
            # Custom training loop with proximal term
            return self._train_proximal(X, y, epochs, batch_size)
        else:
            # Standard training with model.fit (faster)
            hist = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            return np.mean(hist.history['loss'])

    def _train_proximal(self, X, y, epochs, batch_size):
        """Proximal training for unstable clients."""
        n = len(y)
        n_batches = n // batch_size
        epoch_losses = []

        X_tf = tf.constant(X, dtype=tf.float32)
        y_tf = tf.constant(y.reshape(-1, 1), dtype=tf.float32)

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            batch_losses = []

            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                batch_idx = indices[start:end]

                X_batch = tf.gather(X_tf, batch_idx)
                y_batch = tf.gather(y_tf, batch_idx)

                loss = self.proximal_train_step(X_batch, y_batch)
                batch_losses.append(float(loss))

            epoch_losses.append(np.mean(batch_losses))

        return np.mean(epoch_losses)

# ================================================================================
# FedCORA TRAINING
# ================================================================================

def train_fedcora(seed, config, client_data, verbose=True):
    """
    FedCORA: Federated Context-Oriented Reliable Aggregation

    Key innovations:
    1. Adaptive local stabilization (proximal only for unstable clients)
    2. Context-aware fuzzy aggregation
    3. Reduced computation (3 local epochs)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Initialize global model
    global_model = create_model()
    global_weights = get_weights(global_model)

    # History tracking
    history = {
        'round': [], 'val_rmse': [], 'val_loss': [],
        'client_rmse': {'A': [], 'B': [], 'C': []},
        'client_stability': {'A': [], 'B': [], 'C': []},
        'fuzzy_weights': {'A': [], 'B': [], 'C': []},
        'round_times': []
    }

    total_start = time.time()

    for round_num in range(1, config['communication_rounds'] + 1):
        round_start = time.time()

        client_weights_list = []
        client_info = {}
        round_rmse = {}

        # ========== STEP 1: LOCAL TRAINING ==========
        for school in ['A', 'B', 'C']:
            # Create local model
            local_model = create_model()
            set_weights(local_model, global_weights)

            # Evaluate on validation before training to determine stability
            val_metrics = evaluate_model(local_model,
                                         client_data[school]['val']['X'],
                                         client_data[school]['val']['y'])

            # Determine stability
            stability = classify_stability(
                ERI_DATA[school]['eri'],
                val_metrics['RMSE']
            )

            # Adaptive training
            trainer = AdaptiveLocalTrainer(
                local_model,
                config['mu'],
                config['learning_rate'],
                use_proximal=stability['use_proximal']
            )

            if stability['use_proximal']:
                trainer.set_global_weights(global_weights)

            # Train
            avg_loss = trainer.train(
                client_data[school]['train']['X'],
                client_data[school]['train']['y'],
                config['local_epochs'],
                config['batch_size']
            )

            # Evaluate after training
            post_metrics = evaluate_model(local_model,
                                          client_data[school]['val']['X'],
                                          client_data[school]['val']['y'])

            round_rmse[school] = post_metrics['RMSE']
            client_info[school] = {
                'weights': get_weights(local_model),
                'stability': stability,
                'val_rmse': post_metrics['RMSE'],
                'loss': avg_loss
            }

            history['client_rmse'][school].append(post_metrics['RMSE'])
            history['client_stability'][school].append(stability['stable'])

        # ========== STEP 2: FUZZY AGGREGATION ==========
        # Normalize RMSE for fuzzy input (0-1 range, where lower is better)
        max_rmse = 150.0  # Expected max RMSE for normalization
        min_rmse = 40.0   # Expected min RMSE

        fuzzy_inputs = {}
        fuzzy_results = {}

        for school in ['A', 'B', 'C']:
            rmse_norm = (round_rmse[school] - min_rmse) / (max_rmse - min_rmse)
            rmse_norm = np.clip(rmse_norm, 0, 1)

            fuzzy_inputs[school] = {
                'density': SCENARIO[school]['student_density'],
                'ventilation': SCENARIO[school]['ventilation_adequacy'],
                'eri': ERI_DATA[school]['eri'],
                'rmse_normalized': rmse_norm,
                'rmse_ppm': round_rmse[school]
            }

            fuzzy_results[school] = fuzzy_aggregator.compute_weight(
                fuzzy_inputs[school]['density'],
                fuzzy_inputs[school]['ventilation'],
                fuzzy_inputs[school]['eri'],
                rmse_norm
            )

        # Normalize weights to sum = 1
        raw_weights = {s: fuzzy_results[s]['raw_weight'] for s in ['A', 'B', 'C']}
        total_weight = sum(raw_weights.values())
        normalized_weights = {s: w / total_weight for s, w in raw_weights.items()}

        for school in ['A', 'B', 'C']:
            history['fuzzy_weights'][school].append(normalized_weights[school])

        # ========== STEP 3: GLOBAL AGGREGATION ==========
        global_weights = [np.zeros_like(w) for w in client_info['A']['weights']]
        for school in ['A', 'B', 'C']:
            w_school = client_info[school]['weights']
            for i, w in enumerate(w_school):
                global_weights[i] += normalized_weights[school] * w

        set_weights(global_model, global_weights)

        # ========== EVALUATE ==========
        val_metrics = evaluate_model(global_model, X_val_cent, y_val_cent)
        val_loss = global_model.evaluate(X_val_cent, y_val_cent, verbose=0)[0]

        round_time = time.time() - round_start
        history['round'].append(round_num)
        history['val_rmse'].append(val_metrics['RMSE'])
        history['val_loss'].append(val_loss)
        history['round_times'].append(round_time)

        if verbose:
            stability_str = ' '.join([f"{s}:{'S' if client_info[s]['stability']['stable'] else 'U'}"
                                      for s in ['A', 'B', 'C']])
            weights_str = ' '.join([f"{s}:{normalized_weights[s]:.3f}" for s in ['A', 'B', 'C']])
            print(f"  Round {round_num:2d}/{config['communication_rounds']} - "
                  f"val_RMSE: {val_metrics['RMSE']:.2f} | "
                  f"Stability: [{stability_str}] | "
                  f"Weights: [{weights_str}] | "
                  f"time: {round_time:.1f}s")

    total_time = time.time() - total_start

    # Final evaluation
    global_metrics = evaluate_model(global_model, X_test_global, y_test_global)
    per_school = {s: evaluate_model(global_model, client_data[s]['test']['X'],
                                     client_data[s]['test']['y']) for s in ['A', 'B', 'C']}

    return {
        'seed': seed,
        'global_metrics': global_metrics,
        'per_school_metrics': per_school,
        'history': history,
        'total_time_seconds': total_time,
        'avg_round_time': np.mean(history['round_times']),
        'best_val_round': int(np.argmin(history['val_rmse']) + 1),
        'best_val_rmse': float(np.min(history['val_rmse'])),
        'fuzzy_inputs_final': fuzzy_inputs,
        'fuzzy_weights_final': normalized_weights,
        'stability_counts': {
            s: sum(history['client_stability'][s])
            for s in ['A', 'B', 'C']
        }
    }


def train_centralized(seed, config, verbose=True):
    """Train centralized baseline."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = create_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)

    start_time = time.time()
    hist = model.fit(X_train_cent, y_train_cent,
                     validation_data=(X_val_cent, y_val_cent),
                     epochs=100, batch_size=config['batch_size'],
                     callbacks=[early_stop], verbose=0)
    total_time = time.time() - start_time

    best_epoch = np.argmin(hist.history['val_loss']) + 1
    global_metrics = evaluate_model(model, X_test_global, y_test_global)
    per_school = {s: evaluate_model(model, client_data[s]['test']['X'],
                                     client_data[s]['test']['y']) for s in ['A', 'B', 'C']}

    if verbose:
        print(f"  Centralized: RMSE={global_metrics['RMSE']:.2f}, "
              f"epochs={best_epoch}, time={total_time:.1f}s")

    return {
        'seed': seed,
        'global_metrics': global_metrics,
        'per_school_metrics': per_school,
        'total_time_seconds': total_time,
        'best_epoch': best_epoch
    }

# ================================================================================
# RUN EXPERIMENTS
# ================================================================================

print("\n" + "=" * 80)
print("[5] RUNNING EXPERIMENTS")
print("=" * 80)

fedcora_results = []
centralized_results = []

for seed in CONFIG['seeds']:
    print(f"\n{'='*70}")
    print(f"SEED: {seed}")
    print(f"{'='*70}")

    # Centralized baseline
    print("\n[Centralized]")
    cent_result = train_centralized(seed, CONFIG)
    centralized_results.append(cent_result)

    # FedCORA
    print(f"\n[FedCORA]")
    fedcora_result = train_fedcora(seed, CONFIG, client_data)
    fedcora_results.append(fedcora_result)

    print(f"\n  >> FedCORA Final: RMSE={fedcora_result['global_metrics']['RMSE']:.2f} ppm, "
          f"time={fedcora_result['total_time_seconds']:.1f}s")

# ================================================================================
# STATISTICAL ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("[6] STATISTICAL ANALYSIS")
print("=" * 80)

def stats(values):
    return {'mean': np.mean(values), 'std': np.std(values),
            'min': np.min(values), 'max': np.max(values)}

# FedCORA stats
fc_rmse = stats([r['global_metrics']['RMSE'] for r in fedcora_results])
fc_mae = stats([r['global_metrics']['MAE'] for r in fedcora_results])
fc_r2 = stats([r['global_metrics']['R2'] for r in fedcora_results])
fc_time = stats([r['total_time_seconds'] for r in fedcora_results])
fc_round_time = stats([r['avg_round_time'] for r in fedcora_results])

# Centralized stats
cent_rmse = stats([r['global_metrics']['RMSE'] for r in centralized_results])
cent_mae = stats([r['global_metrics']['MAE'] for r in centralized_results])
cent_r2 = stats([r['global_metrics']['R2'] for r in centralized_results])
cent_time = stats([r['total_time_seconds'] for r in centralized_results])

# Per-school stats
fc_school = {}
for school in ['A', 'B', 'C']:
    fc_school[school] = {
        'RMSE': stats([r['per_school_metrics'][school]['RMSE'] for r in fedcora_results]),
        'MAE': stats([r['per_school_metrics'][school]['MAE'] for r in fedcora_results]),
        'R2': stats([r['per_school_metrics'][school]['R2'] for r in fedcora_results]),
    }

print("\n" + "-" * 90)
print(f"{'Method':<15} {'RMSE (ppm)':<20} {'MAE (ppm)':<20} {'R²':<18} {'Time (s)':<15}")
print("-" * 90)
print(f"{'Centralized':<15} {cent_rmse['mean']:>6.2f} ± {cent_rmse['std']:<8.2f}  "
      f"{cent_mae['mean']:>6.2f} ± {cent_mae['std']:<8.2f}  "
      f"{cent_r2['mean']:>6.4f} ± {cent_r2['std']:<8.4f}  "
      f"{cent_time['mean']:>6.1f} ± {cent_time['std']:<5.1f}")
print(f"{'FedCORA':<15} {fc_rmse['mean']:>6.2f} ± {fc_rmse['std']:<8.2f}  "
      f"{fc_mae['mean']:>6.2f} ± {fc_mae['std']:<8.2f}  "
      f"{fc_r2['mean']:>6.4f} ± {fc_r2['std']:<8.4f}  "
      f"{fc_time['mean']:>6.1f} ± {fc_time['std']:<5.1f}")
print("-" * 90)

rmse_gap = fc_rmse['mean'] - cent_rmse['mean']
time_ratio = fc_time['mean'] / cent_time['mean']

print(f"\nGap from Centralized: {rmse_gap:+.2f} ppm ({rmse_gap/cent_rmse['mean']*100:+.1f}%)")
print(f"Time ratio (FedCORA/Centralized): {time_ratio:.2f}x")
print(f"Avg round time: {fc_round_time['mean']:.2f} ± {fc_round_time['std']:.2f} s")

print("\n" + "-" * 75)
print("PER-SCHOOL PERFORMANCE (FedCORA)")
print("-" * 75)
print(f"{'School':<10} {'RMSE (ppm)':<20} {'MAE (ppm)':<20} {'R²':<20}")
print("-" * 75)
for school in ['A', 'B', 'C']:
    s = fc_school[school]
    print(f"{school:<10} {s['RMSE']['mean']:>6.2f} ± {s['RMSE']['std']:<8.2f}  "
          f"{s['MAE']['mean']:>6.2f} ± {s['MAE']['std']:<8.2f}  "
          f"{s['R2']['mean']:>6.4f} ± {s['R2']['std']:<8.4f}")
print("-" * 75)

# ================================================================================
# FUZZY ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("[7] FUZZY INPUT/OUTPUT ANALYSIS")
print("=" * 80)

print("\n--- FUZZY INPUT TABLE ---")
print(f"{'School':<8} {'Density':<10} {'Ventilation':<12} {'ERI':<10} {'Final RMSE':<12}")
print("-" * 52)
for school in ['A', 'B', 'C']:
    final_result = fedcora_results[-1]  # Use last seed for display
    fuzzy_in = final_result['fuzzy_inputs_final'][school]
    print(f"{school:<8} {fuzzy_in['density']:<10.2f} {fuzzy_in['ventilation']:<12.2f} "
          f"{fuzzy_in['eri']:<10.4f} {fuzzy_in['rmse_ppm']:<12.2f}")

print("\n--- STABILITY CLASSIFICATION ---")
for school in ['A', 'B', 'C']:
    total_rounds = CONFIG['communication_rounds']
    stable_rounds = np.mean([r['stability_counts'][school] for r in fedcora_results])
    proximal_activated = total_rounds - stable_rounds
    print(f"School {school}: Stable {stable_rounds:.1f}/{total_rounds} rounds, "
          f"Proximal activated: {proximal_activated:.1f} rounds")

print("\n--- FUZZY OUTPUT (NORMALIZED WEIGHTS) ---")
avg_weights = {s: np.mean([r['history']['fuzzy_weights'][s] for r in fedcora_results], axis=0)
               for s in ['A', 'B', 'C']}
final_avg_weights = {s: avg_weights[s][-1] for s in ['A', 'B', 'C']}

print(f"Final round average weights:")
for school in ['A', 'B', 'C']:
    print(f"  School {school}: {final_avg_weights[school]:.4f}")

# ================================================================================
# CONVERGENCE ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("[8] CONVERGENCE ANALYSIS")
print("=" * 80)

avg_val_rmse = np.mean([r['history']['val_rmse'] for r in fedcora_results], axis=0)
std_val_rmse = np.std([r['history']['val_rmse'] for r in fedcora_results], axis=0)
best_rounds = [r['best_val_round'] for r in fedcora_results]

print(f"Best validation round: {np.mean(best_rounds):.1f} ± {np.std(best_rounds):.1f}")
print(f"Final round RMSE: {avg_val_rmse[-1]:.2f} ± {std_val_rmse[-1]:.2f} ppm")
print(f"Best validation RMSE: {np.min(avg_val_rmse):.2f} ppm (round {np.argmin(avg_val_rmse)+1})")

# ================================================================================
# VISUALIZATION
# ================================================================================

print("\n" + "=" * 80)
print("[9] GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Convergence curve
ax = axes[0, 0]
rounds = range(1, CONFIG['communication_rounds'] + 1)
ax.plot(rounds, avg_val_rmse, 'purple', lw=2, label='FedCORA')
ax.fill_between(rounds, avg_val_rmse - std_val_rmse, avg_val_rmse + std_val_rmse,
                alpha=0.2, color='purple')
ax.axhline(cent_rmse['mean'], color='r', ls='--', lw=2, label='Centralized')
ax.axhspan(cent_rmse['mean'] - cent_rmse['std'], cent_rmse['mean'] + cent_rmse['std'],
           alpha=0.1, color='r')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Validation RMSE (ppm)')
ax.set_title('FedCORA Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Per-seed comparison
ax = axes[0, 1]
seeds_str = [str(s) for s in CONFIG['seeds']]
fc_vals = [r['global_metrics']['RMSE'] for r in fedcora_results]
cent_vals = [r['global_metrics']['RMSE'] for r in centralized_results]
x = np.arange(len(seeds_str))
w = 0.35
ax.bar(x - w/2, cent_vals, w, label='Centralized', color='#e74c3c')
ax.bar(x + w/2, fc_vals, w, label='FedCORA', color='purple')
ax.set_xlabel('Random Seed')
ax.set_ylabel('Test RMSE (ppm)')
ax.set_title('Performance by Seed')
ax.set_xticks(x)
ax.set_xticklabels(seeds_str)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Per-school performance
ax = axes[0, 2]
schools = ['A', 'B', 'C']
school_rmse = [fc_school[s]['RMSE']['mean'] for s in schools]
school_std = [fc_school[s]['RMSE']['std'] for s in schools]
bars = ax.bar(schools, school_rmse, yerr=school_std, capsize=5, color='purple', alpha=0.7)
ax.set_xlabel('School')
ax.set_ylabel('Test RMSE (ppm)')
ax.set_title('FedCORA Per-School Performance')
ax.grid(True, alpha=0.3, axis='y')
for bar, rmse in zip(bars, school_rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{rmse:.1f}', ha='center', fontsize=10)

# 4. Fuzzy weights evolution
ax = axes[1, 0]
for school in ['A', 'B', 'C']:
    avg_w = avg_weights[school]
    ax.plot(rounds, avg_w, label=f'School {school}', lw=2)
ax.axhline(0.333, color='gray', ls=':', label='Equal (1/3)')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Aggregation Weight')
ax.set_title('Fuzzy Aggregation Weights Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Training time comparison
ax = axes[1, 1]
methods = ['Centralized', 'FedCORA']
times = [cent_time['mean'], fc_time['mean']]
time_stds = [cent_time['std'], fc_time['std']]
bars = ax.bar(methods, times, yerr=time_stds, capsize=5, color=['#e74c3c', 'purple'])
ax.set_ylabel('Training Time (seconds)')
ax.set_title('Training Time Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{t:.1f}s', ha='center', fontsize=11)

# 6. Client stability over rounds
ax = axes[1, 2]
for school in ['A', 'B', 'C']:
    stability_avg = np.mean([r['history']['client_stability'][school] for r in fedcora_results], axis=0)
    ax.plot(rounds, stability_avg, label=f'School {school}', lw=2, marker='o', markersize=3)
ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Communication Round')
ax.set_ylabel('Stability (1=Stable, 0=Unstable)')
ax.set_title('Client Stability Classification')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

plt.suptitle(f'FedCORA Evaluation ({len(CONFIG["seeds"])} seeds, {CONFIG["local_epochs"]} local epochs)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fedcora_evaluation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fedcora_evaluation.png")

# ================================================================================
# SAVE RESULTS
# ================================================================================

print("\n" + "=" * 80)
print("[10] SAVING RESULTS")
print("=" * 80)

results_data = {
    'method': 'FedCORA',
    'full_name': 'Federated Context-Oriented Reliable Aggregation',
    'config': CONFIG,
    'scenario': SCENARIO,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

    'summary': {
        'fedcora': {
            'RMSE_mean': fc_rmse['mean'], 'RMSE_std': fc_rmse['std'],
            'MAE_mean': fc_mae['mean'], 'MAE_std': fc_mae['std'],
            'R2_mean': fc_r2['mean'], 'R2_std': fc_r2['std'],
            'time_mean': fc_time['mean'], 'time_std': fc_time['std'],
            'avg_round_time_mean': fc_round_time['mean'], 'avg_round_time_std': fc_round_time['std'],
        },
        'centralized': {
            'RMSE_mean': cent_rmse['mean'], 'RMSE_std': cent_rmse['std'],
            'MAE_mean': cent_mae['mean'], 'MAE_std': cent_mae['std'],
            'R2_mean': cent_r2['mean'], 'R2_std': cent_r2['std'],
            'time_mean': cent_time['mean'], 'time_std': cent_time['std'],
        }
    },

    'per_school': {
        school: {
            'RMSE_mean': fc_school[school]['RMSE']['mean'],
            'RMSE_std': fc_school[school]['RMSE']['std'],
            'MAE_mean': fc_school[school]['MAE']['mean'],
            'MAE_std': fc_school[school]['MAE']['std'],
            'R2_mean': fc_school[school]['R2']['mean'],
            'R2_std': fc_school[school]['R2']['std'],
        }
        for school in ['A', 'B', 'C']
    },

    'eri_data': {school: ERI_DATA[school] for school in ['A', 'B', 'C']},

    'fuzzy_analysis': {
        'final_weights': final_avg_weights,
        'stability_counts': {
            school: float(np.mean([r['stability_counts'][school] for r in fedcora_results]))
            for school in ['A', 'B', 'C']
        }
    },

    'analysis': {
        'rmse_gap_from_centralized': rmse_gap,
        'rmse_gap_percentage': rmse_gap / cent_rmse['mean'] * 100,
        'time_ratio': time_ratio,
        'best_round_mean': float(np.mean(best_rounds)),
        'best_round_std': float(np.std(best_rounds)),
    },

    'convergence': {
        'avg_val_rmse': avg_val_rmse.tolist(),
        'std_val_rmse': std_val_rmse.tolist(),
    },

    'all_runs': [
        {
            'seed': r['seed'],
            'RMSE': r['global_metrics']['RMSE'],
            'MAE': r['global_metrics']['MAE'],
            'R2': r['global_metrics']['R2'],
            'time_seconds': r['total_time_seconds'],
            'best_val_round': r['best_val_round'],
        }
        for r in fedcora_results
    ]
}

with open(os.path.join(OUTPUT_PATH, 'fedcora_results.json'), 'w') as f:
    json.dump(results_data, f, indent=2)
print("Saved: fedcora_results.json")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("[11] FedCORA EVALUATION SUMMARY")
print("=" * 80)

combined_std = np.sqrt(fc_rmse['std']**2 + cent_rmse['std']**2)
significant = abs(rmse_gap) > 2 * combined_std

print(f"""
================================================================================
                         FedCORA EVALUATION COMPLETE
================================================================================

PERFORMANCE SUMMARY:
--------------------
  Method          RMSE (ppm)        MAE (ppm)         R²              Time (s)
  ---------------------------------------------------------------------------
  Centralized     {cent_rmse['mean']:>6.2f} ± {cent_rmse['std']:.2f}      {cent_mae['mean']:>6.2f} ± {cent_mae['std']:.2f}     {cent_r2['mean']:.4f} ± {cent_r2['std']:.4f}   {cent_time['mean']:>6.1f}
  FedCORA         {fc_rmse['mean']:>6.2f} ± {fc_rmse['std']:.2f}      {fc_mae['mean']:>6.2f} ± {fc_mae['std']:.2f}     {fc_r2['mean']:.4f} ± {fc_r2['std']:.4f}   {fc_time['mean']:>6.1f}
  ---------------------------------------------------------------------------

GAP ANALYSIS:
  • RMSE gap from Centralized: {rmse_gap:+.2f} ppm ({rmse_gap/cent_rmse['mean']*100:+.1f}%)
  • Time ratio: {time_ratio:.2f}x

FUZZY AGGREGATION WEIGHTS (Final Round):
  • School A: {final_avg_weights['A']:.4f}
  • School B: {final_avg_weights['B']:.4f}
  • School C: {final_avg_weights['C']:.4f}

ERI VALUES:
  • School A: {ERI_DATA['A']['eri']:.4f}
  • School B: {ERI_DATA['B']['eri']:.4f}
  • School C: {ERI_DATA['C']['eri']:.4f}

CONVERGENCE:
  • Best round: {np.mean(best_rounds):.1f} ± {np.std(best_rounds):.1f}
  • Final RMSE: {avg_val_rmse[-1]:.2f} ± {std_val_rmse[-1]:.2f} ppm

================================================================================
""")

print("FedCORA evaluation complete. Results saved to mission4_fedcora/")
