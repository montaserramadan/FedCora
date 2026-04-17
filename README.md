# FedCORA: Federated Context-Oriented Reliable Aggregation for CO2 Prediction in Smart School Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)

## Author

**Montaser N. A. Ramadan**

- Email: montaser.ramadan@gmail.com
- Google Scholar: [Profile](https://scholar.google.com/citations?user=ixG_9iUAAAAJ&hl=en)

---

## Overview

This repository contains the implementation of **FedCORA** (Federated Context-Oriented Reliable Aggregation), a novel federated learning framework designed for indoor air quality (CO2) prediction in smart school environments.

FedCORA addresses the challenges of **non-IID** (non-independent and identically distributed) data across heterogeneous school environments through three key mechanisms:

1. **Environmental Reliability Index (ERI)** -- A data quality metric derived from variance, missing data ratios, and anomaly detection
2. **Fuzzy Context-Aware Aggregation** -- Domain knowledge-driven weighting using student density, ventilation adequacy, and ERI scores
3. **Adaptive Local Stabilization** -- Selective proximal regularization applied only to unstable clients, reducing computational overhead

The repository also includes implementations of **FedAvg**, **N-FedAvg**, and **FedProx** for comparative evaluation.

---

## Architecture

```
                    +------------------------------------------+
                    |              FedCORA Server               |
                    |                                          |
                    |  +------------+    +-----------------+   |
                    |  |   Fuzzy    |    |    Adaptive     |   |
                    |  |  Context   |--->|   Aggregation   |   |
                    |  | Evaluator  |    |    Engine       |   |
                    |  +------------+    +-----------------+   |
                    |        ^                   |             |
                    +--------|-------------------|-------------+
                             |                   |
                   +---------+---------+---------+---------+
                   |                   |                   |
            +------+------+    +------+------+    +------+------+
            |  School A   |    |  School B   |    |  School C   |
            |  (Client)   |    |  (Client)   |    |  (Client)   |
            |             |    |             |    |             |
            | ERI: 0.928  |    | ERI: 0.810  |    | ERI: 0.754  |
            | Weight: 43% |    | Weight: 38% |    | Weight: 19% |
            | Low density |    | Med density |    | High density|
            | Good vent.  |    | Mod. vent.  |    | Poor vent.  |
            +-------------+    +-------------+    +-------------+
```

---

## Dataset

Real-world CO2 measurements from three schools with distinct environmental characteristics:

| School | Samples | CO2 Range (ppm) | Mean (ppm) | Std (ppm) | Characteristics |
|--------|---------|-----------------|------------|-----------|-----------------|
| A | 13,129 | 418 -- 1,841 | 1,016 | 200 | Low density, good ventilation |
| B | 14,586 | 418 -- 2,275 | 1,280 | 263 | Medium density, moderate ventilation |
| C | 9,703 | 418 -- 2,290 | 1,239 | 451 | High density, poor ventilation |
| **Total** | **37,418** | | | | **Non-IID verified (Kruskal-Wallis p<0.001)** |

Each record contains: `Timestamp`, `CO2 (ppm)`, `Temperature (C)`, `Humidity (%)`

The raw datasets are provided in `data/raw/` as Excel files.

---

## Repository Structure

```
FedCORA/
|
|-- README.md
|-- LICENSE
|-- requirements.txt
|
|-- data/
|   +-- raw/
|       |-- school-A.xlsx
|       |-- school-B.xlsx
|       +-- school-C.xlsx
|
+-- src/
    |-- 01_preprocessing.py              # Data cleaning, feature engineering, scaling
    |-- 01b_preprocessing_corrections.py # Additional preprocessing refinements
    |-- 01c_feature_update.py            # Feature engineering updates
    |-- 02_baseline_models.py            # Local LSTM + Centralized baseline training
    |-- 02b_baseline_analysis.py         # Baseline model analysis
    |-- 03_federated_baselines.py        # FedAvg, N-FedAvg, FedProx implementations
    +-- 04_fedcora.py                    # FedCORA implementation (main contribution)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- TensorFlow 2.15+

### Installation

```bash
git clone https://github.com/dodotik123/Federated-Learning-FedCORA.git
cd Federated-Learning-FedCORA
pip install -r requirements.txt
```

### Running the Experiments

**Step 1: Data Preprocessing**
```bash
python src/01_preprocessing.py
```
Reads raw Excel files from `data/raw/`, performs cleaning, feature engineering, scaling, and generates supervised learning sequences.

**Step 2: Baseline Models**
```bash
python src/02_baseline_models.py
```
Trains local LSTM models per school and a centralized model on combined data.

**Step 3: Federated Learning Baselines**
```bash
python src/03_federated_baselines.py
```
Implements and evaluates FedAvg, N-FedAvg, and FedProx with 30 communication rounds.

**Step 4: FedCORA**
```bash
python src/04_fedcora.py
```
Runs the FedCORA framework with ERI computation, fuzzy aggregation, and adaptive stabilization.

---

## Model Architecture

All methods use the same base LSTM architecture for fair comparison:

```
Input (sequence_length, n_features)
    |
LSTM (64 units)
    |
Dropout (0.2)
    |
Dense (1) --> CO2 prediction
```

### Federated Learning Configuration

| Parameter | FedAvg | N-FedAvg | FedProx | FedCORA |
|-----------|--------|----------|---------|---------|
| Rounds | 30 | 30 | 30 | 30 |
| Local epochs | 5 | 5 | 5 | **3** |
| Batch size | 32 | 32 | 32 | 32 |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 |
| Aggregation | Weighted | Equal | Weighted | **Fuzzy** |
| Proximal term | -- | -- | 0.01 (all) | **0.001 (selective)** |

---

## FedCORA Algorithm

### Environmental Reliability Index (ERI)

```
ERI_k = 0.4 * (1 - variance_k) + 0.3 * (1 - missing_ratio_k) + 0.3 * (1 - anomaly_ratio_k)
```

### Fuzzy Aggregation Weights

Inputs: student density, ventilation adequacy, ERI score

| School | ERI | Fuzzy Weight | Interpretation |
|--------|-----|-------------|----------------|
| A | 0.928 | 42.8% | Highest reliability, stable environment |
| B | 0.810 | 37.9% | Moderate reliability |
| C | 0.754 | 19.4% | Lower reliability, high variance |

### Adaptive Proximal Regularization

Applied selectively to unstable clients only:

```
L_local = L_task + (mu/2) * ||w_local - w_global||^2    (if client is unstable)
L_local = L_task                                          (if client is stable)
```

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{ramadan2026fedcora,
  title={FedCORA: Federated Context-Oriented Reliable Aggregation for CO2 Prediction in Smart School Environments},
  author={Ramadan, Montaser N. A.},
  year={2026},
  url={https://github.com/dodotik123/Federated-Learning-FedCORA}
}
```

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration inquiries, please contact:

**Montaser N. A. Ramadan** -- montaser.ramadan@gmail.com
