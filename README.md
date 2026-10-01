# 🛡️ Dynamic Early Stopping (DES-RL) for Ransomware Detection & Damage Minimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Cybersecurity-AI%20%26%20Malware%20Defense-purple.svg)]()

> **An Actor-Critic Reinforcement Learning framework that dynamically halts ransomware execution mid-stream before destructive file encryption occurs, achieving optimal trade-offs between classification confidence and file-system damage.**

---

## 📌 Executive Summary

Traditional machine learning classifiers evaluate program execution traces either statically (pre-execution) or after observing a fixed number of system calls. In the context of ransomware defense:
- **Waiting too long** guarantees high detection accuracy, but the user suffers catastrophic, irreversible file encryption.
- **Stopping too early** minimizes latency, but risks unacceptable false alarm rates (halting benign system services or user applications).

**DES-RL** formulates runtime ransomware defense as a **sequential decision process (Markov Decision Process)**. An autonomous agent observes real-time API call sequences and dynamically chooses at each discrete time step whether to:
1. `Wait & Observe`: Gather further behavioral evidence while penalizing delay and file-system access.
2. `Classify Benign`: Release the process to continue running normally.
3. `Classify Ransomware`: Terminate the process immediately, arresting file encryption.

---

## 🚀 Key Highlights & Empirical Results

Evaluated across **1,800 test traces** covering benign system activity and five prominent ransomware families (**Cerber, WannaCry, Locky, CryptoLocker, Ryuk**):

* **≥ 99.33% Recall (TPR)** across all ransomware families (100% on Cerber, WannaCry, CryptoLocker, and Ryuk).
* **100.0% True Negative Rate (Benign Accuracy)**, eliminating false stops on legitimate system activities.
* **Near-Zero to Zero File Damage**: Stopped destructive execution prior to payload release in **99.6% - 100%** of cases.
* **Adaptive Latency**: Swiftly stops aggressive threats (WannaCry halted at step 3.4; CryptoLocker halted at step 3.1) while maintaining observation patience for evasive threats (Ryuk observed until step 17.5 before intervention).

### 📊 Multi-Family Performance Benchmark

| Family / Category | Family Characterization | Detection Rate (TPR) | Avg. Stopping Step | Avg. File Damage | Protection Efficiency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Benign** | Normal registry, library loading & temp file cleanup | **100.00%** *(TNR)* | 5.9 | 0.00 | **100.0%** |
| **Cerber** | Ultra-fast execution, heavy write/delete loop | **100.00%** | 4.8 | 0.13 | **99.6%** |
| **WannaCry** | C2 network beaconing & file drops prior to encryption | **100.00%** | 3.4 | 0.00 | **100.0%** |
| **Locky** | Registry-heavy persistence and file hijacking | **99.33%** | 13.2 | 0.00 | **100.0%** |
| **CryptoLocker** | Pre-encryption cryptographic key exchange | **100.00%** | 3.1 | 0.00 | **100.0%** |
| **Ryuk** | Anti-sandbox evasion, system time interrogation | **100.00%** | 17.5 | 0.00 | **100.0%** |

*Protection Efficiency is defined as $1 - \frac{\text{Avg Damage}}{\text{Max Unchecked Damage}}$ where unchecked encryption yields ~35 damaged files per sequence.*

---

## 📈 Visualizations & Empirical Plots

### 1. Detection Rate by Ransomware Family
Consistently high recall across heterogeneous attack profiles, demonstrating robust generalization beyond static signature patterns.

![Detection Rate](family_detection_rate.png)

### 2. Detection Latency vs. File Damage Trade-Off
Illustrates how the RL policy automatically learns family-specific stopping boundaries:

![Stopping Latency vs Damage](family_stopping_latency_vs_damage.png)

### 3. Cumulative Damage Prevention
The agent terminates ransomware processes right as the encryption loop begins, truncating damage curves near zero compared to unchecked baseline execution (dotted lines):

![Damage Prevention Curves](family_damage_prevention_curves.png)

---

## 🧠 System Architecture

The architecture consists of an end-to-end differentiable **Actor-Critic recurrent neural network** coupled with a vectorized early-stopping simulation environment:

```
Telemetry Stream: [ API_1, API_2, ..., API_t ]
         │
         ▼
┌─────────────────────────────────┐
│     Embedding Layer (d=64)      │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Recurrent LSTM (h=128)      │  <── Encodes temporal execution semantics
└─────────────────────────────────┘
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐
│  Policy Head (Actor)  │   │   Value Head (Critic) │
│   Linear -> ReLU      │   │    Linear -> ReLU     │
│    -> Logits (3)      │   │      -> V(s) (1)      │
└───────────────────────┘   └───────────────────────┘
    │                               │
    ▼                               ▼
Action Probability Distribution    Expected Return Estimate
[ Wait, Benign, Ransomware ]
```

### Reward Formulation

At each step $t$, given state $s_t$ and action $a_t$:
- **Action 0 (`Wait & Observe`)**:
  $$R_t = -\alpha - \beta \cdot \text{Damage}_t$$
  *(Penalizes elapsed computation time $\alpha$ and any intermediate file write/delete operations $\beta$)*
- **Action 1 (`Classify Benign`)**:
  $$R_{\text{term}} = \begin{cases} +\gamma & \text{if True Benign} \\ -\text{Penalty}_{FN} & \text{if True Ransomware (Misclassification)} \end{cases}$$
- **Action 2 (`Classify Ransomware`)**:
  $$R_{\text{term}} = \begin{cases} +\gamma - \beta \sum_{\tau=0}^{t}\text{Damage}_\tau & \text{if True Ransomware} \\ -\text{Penalty}_{FP} & \text{if True Benign (False Alarm)} \end{cases}$$

---

## 📂 Repository Structure

```
.
├── DATA/
│   ├── data_loader.py              # Sequence generator with realistic API overlap & noise
│   ├── patch_data_loader.py        # Noise and decoy injection utility
│   └── data_loader_backup.py       # Baseline reference data generator
├── MODEL/
│   ├── models.py                   # DESModel definition (Embedding + LSTM + Dual Heads)
│   └── des_model.pt                # Trained PyTorch model checkpoint
├── TRAIN/
│   └── train.py                    # Vectorized Actor-Critic RL training pipeline
├── EVAL/
│   ├── evaluate.py                 # Benchmarking against static-step stopping baselines
│   └── evaluate_families.py        # Multi-family validation and matplotlib figure generation
├── family_detection_rate.png       # Detection recall per ransomware family
├── family_stopping_latency_vs_damage.png  # Trade-off scatter plot
├── family_damage_prevention_curves.png    # Step-by-step blocked damage curves
├── results_comparison.png          # Comparison vs static observation windows
└── README.md                       # Project documentation
```

---

## 🛠️ Quickstart & Usage

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/YashsTiwari/Used-Car-Price-Prediction.git
cd Used-Car-Price-Prediction
pip install torch numpy scikit-learn matplotlib
```

### 2. Training the DES-RL Agent
Train the policy network from scratch using policy gradients and value baseline:
```bash
python -m TRAIN.train
```
*Trained weights will automatically be checkpointed to `MODEL/des_model.pt`.*

### 3. Evaluating Multi-Family Performance & Generating Figures
Run the full empirical evaluation suite across simulated ransomware families:
```bash
python -m EVAL.evaluate_families
```

### 4. Evaluating Against Static Stopping Baselines
Compare the dynamic stopping agent against fixed-observation baselines (e.g., stopping at step 10, 20, ..., 90):
```bash
python -m EVAL.evaluate
```

---

## 🔬 Scientific & Practical Significance

1. **Overcoming the "All-or-Nothing" Dilemma**: Standard detectors need complete sandbox logs. DES-RL makes rapid real-time decisions within the first 3–18 API calls.
2. **Robust Against Evasion & Decoy Calls**: With synthetic decoy APIs injected (e.g., benign registry queries before encryption), the recurrent state representation effectively separates camouflage from true attack trajectories.
3. **Generalization Across Diverse Families**: Without hardcoded heuristics, the model autonomously learned distinct stopping thresholds tailored to the speed of Cerber, the network signatures of WannaCry/CryptoLocker, and the deliberate delays of Ryuk.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
