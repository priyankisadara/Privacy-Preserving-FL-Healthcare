"""
Main Orchestration Script for Privacy-Preserving Federated Learning.
Executes multi-client federated simulation across healthcare institution nodes.
Supports FedAvg, FedProx, Differential Privacy (DP-SGD), and Secure Aggregation (SecAgg).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import json
import yaml
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.download_datasets import download_medmnist, get_labels_from_dataset, compute_class_weights
from data.partitioner import partition_iid, partition_dirichlet, analyze_client_distribution
from src.models.cnn_pneumonia import get_model
from src.client.dp_trainer import LocalTrainer
from src.security.secure_aggregation import SecureAggregator
from src.evaluation.metrics import evaluate_model


def run_federated_simulation(config_path: str = "configs/default_config.yaml", override_args = None):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if override_args:
        if override_args.rounds is not None:
            config["federation"]["num_rounds"] = override_args.rounds
        if override_args.clients is not None:
            config["federation"]["num_clients"] = override_args.clients
        if override_args.alpha is not None:
            config["distribution"]["dirichlet_alpha"] = override_args.alpha
        if override_args.dp is not None:
            config["differential_privacy"]["enabled"] = override_args.dp

    print("\n" + "=" * 75)
    print("      PRIVACY-PRESERVING FEDERATED LEARNING FOR HEALTHCARE")
    print("=" * 75)
    print(f"[*] Simulation Parameters:")
    print(f"    - Healthcare Benchmark:  {config['dataset']['name']} (Chest X-Ray Pneumonia)")
    print(f"    - Participating Nodes:   {config['federation']['num_clients']} Hospital Centers")
    print(f"    - Communication Rounds:  {config['federation']['num_rounds']}")
    print(f"    - Aggregation Strategy:  {config['federation']['strategy']}")
    print(f"    - Data Distribution:     {config['distribution']['type']} (alpha={config['distribution']['dirichlet_alpha']})")
    print(f"    - Differential Privacy:  {config['differential_privacy']['enabled']}")
    if config["differential_privacy"]["enabled"]:
        print(f"      (Target ε={config['differential_privacy']['target_epsilon']}, δ={config['differential_privacy']['target_delta']}, Noise σ={config['differential_privacy']['noise_multiplier']})")
    print(f"    - Secure Aggregation:    {config['secure_aggregation']['enabled']}")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")

    # 2. Download and Load Data
    train_ds, val_ds, test_ds = download_medmnist(
        dataset_name=config["dataset"]["name"], data_dir=config["dataset"]["data_dir"]
    )
    train_labels = get_labels_from_dataset(train_ds)
    class_weights = compute_class_weights(train_labels)
    print(f"[*] Balanced Class Weights: Normal={class_weights[0]:.2f}, Pneumonia={class_weights[1]:.2f}")
    num_clients = config["federation"]["num_clients"]

    # 3. Partition Data across Hospital Nodes
    if config["distribution"]["type"] == "iid":
        client_subsets = partition_iid(train_ds, num_clients=num_clients)
    else:
        alpha = float(config["distribution"]["dirichlet_alpha"])
        client_indices = partition_dirichlet(train_labels, num_clients=num_clients, alpha=alpha)
        client_subsets = {cid: Subset(train_ds, idxs) for cid, idxs in client_indices.items()}
        analyze_client_distribution(client_indices, train_labels)

    test_loader = DataLoader(
        test_ds, batch_size=config["local_training"]["batch_size"], shuffle=False
    )

    # 4. Initialize Global Model
    global_model = get_model().to(device)
    sec_agg = SecureAggregator(num_clients=num_clients) if config["secure_aggregation"]["enabled"] else None

    # Track metrics per round
    history = []

    # Initial Baseline Evaluation (Round 0)
    print("\n[*] Evaluating Initial Global Model (Round 0)...")
    init_eval = evaluate_model(global_model, test_loader, device)
    print(f"    -> Initial Test Accuracy: {init_eval['accuracy'] * 100:.2f}% | Loss: {init_eval['loss']:.4f} | AUROC: {init_eval['auroc']:.4f}")
    history.append({
        "round": 0,
        "test_loss": init_eval["loss"],
        "test_accuracy": init_eval["accuracy"],
        "test_auroc": init_eval["auroc"],
        "test_f1": init_eval["f1"],
        "train_loss": 0.0,
        "epsilon": 0.0
    })

    # 5. Federated Communication Rounds
    num_rounds = config["federation"]["num_rounds"]
    for round_num in range(1, num_rounds + 1):
        print(f"\n>>> Commencing Communication Round {round_num:02d} / {num_rounds:02d} <<<")
        
        # Broadcast global parameters
        global_params_tensors = [p.clone().detach() for p in global_model.parameters()]
        global_state_dict = copy.deepcopy(global_model.state_dict())

        client_updates = []
        client_sample_counts = []
        client_train_losses = []
        client_epsilons = []

        # Local hospital training loops
        for cid in range(num_clients):
            n_samples = len(client_subsets[cid])
            if n_samples == 0:
                print(f"    [!] Warning: Hospital {cid} has 0 samples; skipping.")
                continue

            # Create local model instance initialized with global weights
            local_model = get_model().to(device)
            local_model.load_state_dict(global_state_dict)

            local_batch_size = max(1, min(config["local_training"]["batch_size"], n_samples))
            train_loader = DataLoader(
                client_subsets[cid],
                batch_size=local_batch_size,
                shuffle=True,
                drop_last=False
            )

            # Cosine / exponential round-level learning rate schedule
            round_lr = config["local_training"]["learning_rate"] * (0.97 ** (round_num - 1))

            trainer = LocalTrainer(
                model=local_model,
                device=device,
                learning_rate=round_lr,
                weight_decay=config["local_training"]["weight_decay"],
                optimizer_type=config["local_training"].get("optimizer", "Adam"),
                class_weights=class_weights,
                dp_enabled=config["differential_privacy"]["enabled"],
                target_epsilon=config["differential_privacy"]["target_epsilon"],
                target_delta=config["differential_privacy"]["target_delta"],
                max_grad_norm=config["differential_privacy"]["max_grad_norm"],
                noise_multiplier=config["differential_privacy"]["noise_multiplier"],
                fedprox_mu=config["federation"]["fedprox_mu"] if config["federation"]["strategy"] == "FedProx" else 0.0
            )

            loss, eps = trainer.train_epoch(
                train_loader=train_loader,
                epochs=config["local_training"]["epochs"],
                global_model_params=global_params_tensors
            )

            local_weights = [val.cpu().numpy() for _, val in local_model.state_dict().items()]
            n_samples = len(client_subsets[cid])

            client_updates.append(local_weights)
            client_sample_counts.append(n_samples)
            client_train_losses.append(loss)
            client_epsilons.append(eps)

        avg_train_loss = np.mean(client_train_losses)
        max_epsilon = np.max(client_epsilons)

        # 6. Secure Aggregation & Model Weight Aggregation
        total_samples = sum(client_sample_counts)
        aggregated_weights = []
        num_layers = len(client_updates[0])

        if sec_agg is not None:
            weight_shapes = [w.shape for w in client_updates[0]]
            pairwise_masks = sec_agg.generate_pairwise_masks(weight_shapes)
            assert sec_agg.verify_cancellation(pairwise_masks), "SecAgg mask verification failed!"
            # Clients mask their weighted models: (n_k / N) * w_k + mask_k
            masked_updates = [
                sec_agg.mask_weighted_weights(
                    cid, client_updates[cid], client_sample_counts[cid] / total_samples, pairwise_masks
                )
                for cid in range(num_clients)
            ]
            # Server sums masked updates: masks cancel identically to 0, yielding exact weighted average
            for l in range(num_layers):
                layer_sum = sum(masked_updates[cid][l] for cid in range(num_clients))
                aggregated_weights.append(layer_sum)
        else:
            for l in range(num_layers):
                layer_sum = np.zeros_like(client_updates[0][l], dtype=np.float32)
                for cid in range(num_clients):
                    weight_factor = client_sample_counts[cid] / total_samples
                    layer_sum += weight_factor * client_updates[cid][l]
                aggregated_weights.append(layer_sum)

        # Update global model state dict
        new_state_dict = OrderedDict()
        for idx, key in enumerate(global_model.state_dict().keys()):
            new_state_dict[key] = torch.tensor(aggregated_weights[idx])
        global_model.load_state_dict(new_state_dict)

        # 8. Evaluate New Global Model on Central Test Set
        eval_res = evaluate_model(global_model, test_loader, device)

        print(f"--- [Round {round_num:02d} Summary] ---")
        print(f"    Train Loss (Avg): {avg_train_loss:.4f}")
        print(f"    Test Loss:        {eval_res['loss']:.4f}")
        print(f"    Test Accuracy:    {eval_res['accuracy'] * 100:.2f}% (Opt: {eval_res.get('optimal_accuracy', eval_res['accuracy']) * 100:.2f}%)")
        print(f"    Test AUROC:       {eval_res['auroc']:.4f}")
        print(f"    Test Macro F1:    {eval_res['f1']:.4f} (Opt F1: {eval_res.get('optimal_f1', eval_res['f1']):.4f} @ thr={eval_res.get('optimal_threshold', 0.5):.2f})")
        print(f"    Test Precision:   {eval_res['precision']:.4f} | Recall: {eval_res['recall']:.4f}")
        if config["differential_privacy"]["enabled"]:
            print(f"    Cumulative Privacy Loss: ε = {max_epsilon:.2f} (δ = {config['differential_privacy']['target_delta']})")

        history.append({
            "round": round_num,
            "train_loss": float(avg_train_loss),
            "test_loss": float(eval_res["loss"]),
            "test_accuracy": float(eval_res["accuracy"]),
            "test_auroc": float(eval_res["auroc"]),
            "test_f1": float(eval_res["f1"]),
            "test_precision": float(eval_res["precision"]),
            "test_recall": float(eval_res["recall"]),
            "optimal_threshold": float(eval_res.get("optimal_threshold", 0.5)),
            "optimal_accuracy": float(eval_res.get("optimal_accuracy", eval_res["accuracy"])),
            "optimal_f1": float(eval_res.get("optimal_f1", eval_res["f1"])),
            "epsilon": float(max_epsilon)
        })

    # 9. Save Results
    os.makedirs("experiments/results", exist_ok=True)
    dp_tag = "dp" if config["differential_privacy"]["enabled"] else "nodp"
    strat_tag = config["federation"]["strategy"].lower()
    out_file = f"experiments/results/fl_history_{strat_tag}_{dp_tag}.json"
    with open(out_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\n[+] Simulation Complete! Results logged to: '{out_file}'")

    # Save final model weights
    model_save_path = f"experiments/results/global_model_{strat_tag}_{dp_tag}.pt"
    torch.save(global_model.state_dict(), model_save_path)
    print(f"[+] Global Model Checkpoint saved to: '{model_save_path}'")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy-Preserving Federated Learning Runner")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Config file path")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--clients", type=int, default=None, help="Override number of clients")
    parser.add_argument("--alpha", type=float, default=None, help="Override Dirichlet alpha")
    parser.add_argument("--dp", action="store_true", default=None, help="Enable Differential Privacy")
    parser.add_argument("--no-dp", dest="dp", action="store_false", help="Disable Differential Privacy")
    args = parser.parse_args()

    run_federated_simulation(config_path=args.config, override_args=args)