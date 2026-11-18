"""
Custom Federated Aggregation Strategies (FedAvg & FedProx) with
Centralized Global Evaluation & Comprehensive Metrics Logging.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.cnn_pneumonia import get_model
from src.evaluation.metrics import evaluate_model


def weighted_metrics_aggregator(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Computes weighted average of client metrics based on number of samples.
    """
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}

    aggregated = {}
    for key in metrics[0][1].keys():
        if isinstance(metrics[0][1][key], (int, float)):
            weighted_val = sum([num_examples * m[key] for num_examples, m in metrics])
            aggregated[key] = weighted_val / total_examples

    return aggregated


def get_evaluate_fn(
    test_loader: DataLoader, device: torch.device, log_history: List[Dict]
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Returns a server-side evaluation function to benchmark the global model
    on the held-out central test dataset after every communication round.
    """
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        model = get_model()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        metrics = evaluate_model(model, test_loader, device)

        # Store in round log
        round_log = {
            "round": server_round,
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "test_auroc": metrics["auroc"],
            "test_f1": metrics["f1"],
        }
        log_history.append(round_log)

        print(f"\n==================== [ROUND {server_round:02d} GLOBAL TEST RESULTS] ====================")
        print(f"  Test Loss:     {metrics['loss']:.4f}")
        print(f"  Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"  Test AUROC:    {metrics['auroc']:.4f}")
        print(f"  Test Macro F1: {metrics['f1']:.4f}")
        print("========================================================================\n")

        return metrics["loss"], metrics

    return evaluate

