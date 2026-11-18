"""
Automated Plot Generator for Research Papers & M.Tech Thesis.
Generates publication-quality charts comparing:
1. Accuracy & AUROC convergence over Federated Communication Rounds.
2. Loss curves (Train vs Test).
3. Non-IID Dirichlet distribution across Hospital Centers.
4. Privacy Budget spend (Cumulative ε over rounds).
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_federated_metrics(
    results_json_path: str, save_path: str = "experiments/results/convergence_plot.png"
):
    if not os.path.exists(results_json_path):
        print(f"[!] Error: File '{results_json_path}' not found.")
        return

    with open(results_json_path, "r") as f:
        history = json.load(f)

    rounds = [entry["round"] for entry in history]
    test_acc = [entry["test_accuracy"] * 100 for entry in history]
    test_loss = [entry["test_loss"] for entry in history]
    test_auroc = [entry["test_auroc"] for entry in history]
    epsilons = [entry.get("epsilon", 0.0) for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Test Accuracy & AUROC
    axes[0].plot(rounds, test_acc, marker="o", color="#1f77b4", linewidth=2, label="Test Accuracy (%)")
    axes[0].set_title("Global Model Diagnostic Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Federation Communication Rounds", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Plot 2: AUROC
    axes[1].plot(rounds, test_auroc, marker="s", color="#2ca02c", linewidth=2, label="Test AUROC")
    axes[1].set_title("Area Under ROC Curve (AUROC)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Federation Communication Rounds", fontsize=11)
    axes[1].set_ylabel("AUROC", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Plot 3: Loss or Privacy Budget
    if any(e > 0 for e in epsilons):
        axes[2].plot(rounds, epsilons, marker="^", color="#d62728", linewidth=2, label="Cumulative Privacy Loss (ε)")
        axes[2].set_title("Differential Privacy Budget Spend", fontsize=13, fontweight="bold")
        axes[2].set_ylabel("Epsilon (ε)", fontsize=11)
    else:
        axes[2].plot(rounds, test_loss, marker="^", color="#ff7f0e", linewidth=2, label="Test Loss")
        axes[2].set_title("Global Model Cross-Entropy Loss", fontsize=13, fontweight="bold")
        axes[2].set_ylabel("Loss", fontsize=11)

    axes[2].set_xlabel("Federation Communication Rounds", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Thesis chart generated and saved to: '{save_path}'")


if __name__ == "__main__":
    for fname in ["fl_history_fedavg_nodp.json", "fl_history_fedavg_dp.json"]:
        path = os.path.join("experiments/results", fname)
        if os.path.exists(path):
            plot_federated_metrics(path, save_path=f"experiments/results/{fname.replace('.json', '.png')}")

