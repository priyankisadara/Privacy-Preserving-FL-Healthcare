"""
Interactive Streamlit Dashboard for Privacy-Preserving Federated Learning in Healthcare.
Visualizes:
1. Heterogeneous Hospital Node Distributions (Dirichlet alpha)
2. Global Model Convergence (Accuracy, AUROC, Loss)
3. Differential Privacy Budget (Epsilon, Delta)
4. Live Diagnostic Inference on Test Patient Chest X-Rays
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.models.cnn_pneumonia import get_model
from data.download_datasets import download_medmnist, get_labels_from_dataset
from data.partitioner import partition_dirichlet

# Streamlit Page Setup
st.set_page_config(
    page_title="PPFL Healthcare Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Privacy-Preserving Federated Learning for Healthcare")
st.markdown(
    "A decentralized clinical machine learning framework evaluated on **MedMNIST (Pneumonia Detection)** "
    "with **Differential Privacy (DP-SGD)** and **Secure Aggregation (SecAgg)**."
)

# Sidebar Controls
st.sidebar.header("🔬 Experiment Parameters")
experiment_mode = st.sidebar.selectbox(
    "Select Experiment Run",
    ["FedAvg Baseline (No DP)", "FedAvg with Differential Privacy (DP-SGD)"],
    index=0
)

num_clients = st.sidebar.slider("Participating Hospital Nodes (K)", min_value=2, max_value=10, value=5)
dirichlet_alpha = st.sidebar.slider(
    "Dirichlet Skewness (α)",
    min_value=0.1, max_value=5.0, value=0.5, step=0.1,
    help="Lower α (<1.0) means high demographic/prevalence skew between hospitals; Higher α is near-IID."
)

# Load Selected Run History
if "No DP" in experiment_mode:
    log_file = "experiments/results/fl_history_fedavg_nodp.json"
    model_file = "experiments/results/global_model_fedavg_nodp.pt"
    dp_status = "Disabled"
else:
    log_file = "experiments/results/fl_history_fedavg_dp.json"
    model_file = "experiments/results/global_model_fedavg_dp.pt"
    dp_status = "Enabled (ε ≤ 5.0, δ = 10⁻⁵)"

# Metrics Overview Cards
col1, col2, col3, col4 = st.columns(4)
history = []
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        history = json.load(f)

latest = history[-1] if history else {"test_accuracy": 0.0, "test_auroc": 0.0, "test_loss": 0.0, "epsilon": 0.0}

with col1:
    st.metric(label="Global Test Accuracy", value=f"{latest['test_accuracy'] * 100:.2f}%")
with col2:
    st.metric(label="Global AUROC", value=f"{latest['test_auroc']:.4f}")
with col3:
    st.metric(label="Privacy Budget (ε)", value=f"{latest.get('epsilon', 0.0):.2f}" if dp_status != "Disabled" else "N/A (No DP)")
with col4:
    st.metric(label="Federation Status", value="SecAgg Active")

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Hospital Distribution (Non-IID)",
    "📈 Federation Convergence & Trade-Offs",
    "🩺 Live Patient Diagnostic Test"
])

# TAB 1: Hospital Distribution
with tab1:
    st.subheader("Simulated Patient Distribution across Hospital Nodes")
    st.markdown(
        f"Shows label distribution among **{num_clients} hospitals** under **Dirichlet $\\alpha = {dirichlet_alpha}$**."
    )
    
    train_ds, _, _ = download_medmnist(data_dir="./data/raw")
    labels = get_labels_from_dataset(train_ds)
    client_indices = partition_dirichlet(labels, num_clients=num_clients, alpha=dirichlet_alpha)

    # Build DataFrame
    dist_data = []
    for cid, idxs in client_indices.items():
        c_labels = labels[idxs]
        norm_count = int(np.sum(c_labels == 0))
        pneu_count = int(np.sum(c_labels == 1))
        dist_data.append({
            "Hospital": f"Hospital {cid + 1}",
            "Normal Cases (Class 0)": norm_count,
            "Pneumonia Cases (Class 1)": pneu_count,
            "Total Patient Records": len(idxs)
        })
    df_dist = pd.DataFrame(dist_data)
    
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.dataframe(df_dist.set_index("Hospital"), use_container_width=True)
    with col_r:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_dist.set_index("Hospital")[["Normal Cases (Class 0)", "Pneumonia Cases (Class 1)"]].plot(
            kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728"]
        )
        ax.set_ylabel("Number of Patients")
        ax.set_title(f"Clinical Case Imbalance across Hospitals (α={dirichlet_alpha})")
        plt.xticks(rotation=0)
        st.pyplot(fig)

# TAB 2: Training Convergence
with tab2:
    st.subheader("Federated Learning Rounds & Privacy Budget Tracking")
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["Accuracy (%)"] = df_hist["test_accuracy"] * 100
        
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df_hist.set_index("round")[["Accuracy (%)"]])
        with c2:
            st.line_chart(df_hist.set_index("round")[["test_auroc", "test_loss"]])
            
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No experiment logs found for this mode yet. Run `python experiments/run_federated.py` to generate logs.")

# TAB 3: Diagnostic Inference
with tab3:
    st.subheader("Test Trained Global Model on Sample Patient Chest X-Rays")
    
    if os.path.exists(model_file):
        device = torch.device("cpu")
        model = get_model().to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        _, _, test_ds = download_medmnist(data_dir="./data/raw")
        
        test_idx = st.slider("Select Patient Sample Index (from central test set)", 0, len(test_ds) - 1, 15)
        img_tensor, ground_truth = test_ds[test_idx]
        
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1).squeeze().numpy()
            pred_class = int(np.argmax(probs))

        classes = ["Normal", "Pneumonia"]
        gt_name = classes[int(ground_truth[0])]
        pred_name = classes[pred_class]
        confidence = probs[pred_class] * 100

        col_img, col_res = st.columns([1, 2])
        with col_img:
            # Display image (28x28 grayscale)
            img_np = img_tensor.squeeze().numpy()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_np, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Patient Record #{test_idx}")
            st.pyplot(fig)

        with col_res:
            st.markdown(f"### Diagnosis Prediction: **{pred_name}**")
            st.markdown(f"- **Ground Truth Diagnosis:** `{gt_name}`")
            st.markdown(f"- **Model Diagnostic Confidence:** `{confidence:.2f}%`")
            st.progress(float(probs[1]))
            st.caption(f"Calculated Probability of Pneumonia: {probs[1] * 100:.2f}%")
            
            if pred_class == ground_truth[0]:
                st.success("✅ Global model prediction matches clinical ground truth!")
            else:
                st.warning("⚠️ Misclassification on this challenging sample.")
    else:
        st.warning(f"Checkpoint file `{model_file}` not found. Run the federated simulation to train the global model.")

