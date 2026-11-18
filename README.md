# Privacy-Preserving Federated Learning for Healthcare

An end-to-end framework enabling multiple healthcare institutions (simulated hospitals) to collaboratively train diagnostic deep learning models on sensitive patient records without sharing local raw data.

---

## 🌟 Key Features

1. **Decentralized Multi-Hospital Simulation:**
   - Evaluated on **MedMNIST (PneumoniaMNIST)** pediatric chest X-rays.
   - Simulates real-world clinical demographic skew using **Dirichlet Non-IID Partitioning ($\alpha$)**.
2. **Differential Privacy (DP-SGD via Opacus):**
   - Per-sample gradient clipping ($C$) and calibrated Gaussian noise injection ($\sigma$).
   - Formal mathematical privacy guarantees tracked with Rényi Differential Privacy ($(\epsilon, \delta)$ budget).
3. **Secure Aggregation (SecAgg):**
   - Pairwise zero-sum random secret masking ensures the central aggregator only learns the sum of weights, never individual client models.
4. **Interactive Streamlit Dashboard:**
   - Dynamic hospital distribution visualizer, real-time round convergence graphs, and live patient diagnostic inference.

---

## 📁 Project Structure

```text
Privacy-Preserving-FL-Healthcare/
│
├── configs/
│   └── default_config.yaml         # Hyperparameters (clients, rounds, DP, Dirichlet alpha)
├── data/
│   ├── download_datasets.py        # MedMNIST downloader & preprocessor
│   └── partitioner.py              # IID and Non-IID Dirichlet partitioner
├── src/
│   ├── models/
│   │   └── cnn_pneumonia.py        # DP-compatible CNN with GroupNorm
│   ├── client/
│   │   ├── dp_trainer.py           # Opacus DP-SGD & FedProx local trainer
│   │   └── fl_client.py            # Flower NumPyClient wrapper
│   ├── server/
│   │   └── strategies.py           # FedAvg / FedProx strategies & evaluation
│   ├── security/
│   │   └── secure_aggregation.py   # Pairwise zero-sum secret masking
│   └── evaluation/
│       └── metrics.py              # Accuracy, AUROC, Macro F1, Loss
├── experiments/
│   ├── run_federated.py            # Main federated simulation runner
│   ├── plot_results.py             # Publication & thesis graph generator
│   └── results/                    # Output checkpoints & JSON metrics
├── app/
│   └── dashboard.py                # Interactive Streamlit dashboard
├── IMPLEMENTATION_PLAN.md          # Full M.Tech research roadmap
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
Ensure your virtual environment is active:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run Federated Learning Simulation

**Standard Federated Learning (FedAvg):**
```powershell
python experiments/run_federated.py --rounds 10 --no-dp
```

**Privacy-Preserving Federated Learning (FedAvg + DP-SGD + SecAgg):**
```powershell
python experiments/run_federated.py --rounds 10 --dp
```

**Custom Skewness (Dirichlet $\alpha$):**
```powershell
# Extreme non-IID skew (alpha=0.1) across 5 hospitals:
python experiments/run_federated.py --rounds 10 --alpha 0.1 --dp
```

### 3. Generate Research Thesis Plots
```powershell
python experiments/plot_results.py
```
*(High-resolution 300 DPI figures will be saved in `experiments/results/`)*

### 4. Launch the Interactive Dashboard
```powershell
streamlit run app/dashboard.py
```
*(Opens an interactive web interface at `http://localhost:8501`)*

---

## 📊 Experimental Results

Extensive empirical evaluations were conducted on the **MedMNIST PneumoniaMNIST** dataset across 5 simulated hospital nodes under Dirichlet non-IID class skew ($\alpha = 0.5$):

| Configuration | Test Accuracy | AUROC | Macro F1 | Differential Privacy $(\epsilon, \delta)$ | Secure Aggregation |
|---|:---:|:---:|:---:|:---:|:---:|
| **Standard FL (GroupNorm + Augmentations)** | **86.86%** | **0.9273** | **0.8603** | $\infty$ (No DP) | Verified |
| **PPFL (+ DP-SGD $\sigma=0.8, C=1.5$)** | **70.67%** | **0.7306** | **0.6964** | $\epsilon \le 5.07, \delta=10^{-5}$ | Verified |
| Baseline (with Mask Cancellation Anomaly) | 55.45% | 0.6313 | 0.5488 | $\infty$ (No DP) | Corrupted |

> **Key Discovery:** Standard pairwise SecAgg fails under sample-weighted federated averaging ($\sum \frac{n_k}{N} w_k$). Our framework establishes a pre-weighted zero-sum masking formulation that eliminates residual noise artifacts, boosting diagnostic accuracy from ~55% to **86.86%**.

---

## 📊 Evaluation Metrics
- **Accuracy & AUROC:** Model diagnostic performance on unseen central test patient records.
- **Privacy Budget ($\epsilon, \delta$):** Bounded privacy risk calculated through moments accountant.
- **Client Drift Resilience:** Performance stability across varying degrees of Non-IID skew ($\alpha$).

