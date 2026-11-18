"""
Flower Federated Learning Client for Healthcare Institution Simulation.
Encapsulates local data loading, training with DP-SGD / FedProx, and SecAgg masking.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.cnn_pneumonia import get_model
from src.client.dp_trainer import LocalTrainer
from src.evaluation.metrics import evaluate_model


class HealthcareClient(fl.client.NumPyClient):
    """
    Simulates a single Hospital / Clinical Diagnostic node in Federated Learning.
    """
    def __init__(
        self,
        client_id: int,
        train_subset: Subset,
        val_subset: Subset,
        batch_size: int = 32,
        epochs: int = 3,
        lr: float = 0.001,
        dp_enabled: bool = False,
        target_epsilon: float = 5.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        fedprox_mu: float = 0.0,
        device: torch.device = None
    ):
        self.client_id = client_id
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dp_enabled = dp_enabled
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.fedprox_mu = fedprox_mu
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_subset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        self.val_loader = DataLoader(
            self.val_subset, batch_size=self.batch_size, shuffle=False
        )

        self.model = get_model()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Returns model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Sets model weights from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Executes local training round at this hospital node.
        """
        self.set_parameters(parameters)

        # Cache global weights for FedProx penalty if needed
        global_params = [torch.tensor(p).clone().detach() for p in parameters]

        trainer = LocalTrainer(
            model=self.model,
            device=self.device,
            learning_rate=self.lr,
            dp_enabled=self.dp_enabled,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
            fedprox_mu=self.fedprox_mu,
        )

        loss, epsilon = trainer.train_epoch(
            self.train_loader, epochs=self.epochs, global_model_params=global_params
        )

        updated_parameters = self.get_parameters(config={})
        num_samples = len(self.train_subset)

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(loss),
            "epsilon": float(epsilon),
            "num_samples": num_samples
        }

        return updated_parameters, num_samples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """
        Evaluates current global model on local hospital validation data.
        """
        self.set_parameters(parameters)
        eval_metrics = evaluate_model(self.model, self.val_loader, self.device)
        return eval_metrics["loss"], len(self.val_subset), eval_metrics

