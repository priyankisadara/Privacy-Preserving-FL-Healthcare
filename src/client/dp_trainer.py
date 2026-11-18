"""
Local Training Engine with Differential Privacy (DP-SGD), Class Weighting,
and Proximal Regularization (FedProx).
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from typing import Dict, Tuple, Optional


class LocalTrainer:
    """
    Handles local client training with optional Differential Privacy (DP-SGD),
    balanced class weights, and FedProx proximal drift penalty.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer_type: str = "Adam",
        class_weights: Optional[torch.Tensor] = None,
        dp_enabled: bool = False,
        target_epsilon: float = 5.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        fedprox_mu: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.dp_enabled = dp_enabled
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.fedprox_mu = fedprox_mu
        self.privacy_engine: Optional[PrivacyEngine] = None

    def train_epoch(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        global_model_params: Optional[list] = None
    ) -> Tuple[float, float]:
        """
        Trains model locally for `epochs`.
        Returns: (average_loss, current_epsilon)
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        if self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        current_epsilon = 0.0
        active_loader = train_loader

        if self.dp_enabled:
            # Attach Opacus PrivacyEngine
            self.privacy_engine = PrivacyEngine()
            self.model, optimizer, active_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )

        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for images, targets in active_loader:
                images = images.to(self.device)
                targets = targets.squeeze().long().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)

                # Add FedProx proximal penalty if mu > 0
                if self.fedprox_mu > 0.0 and global_model_params is not None:
                    proximal_term = 0.0
                    for p, g_p in zip(self.model.parameters(), global_model_params):
                        proximal_term += torch.sum((p - g_p.to(self.device)) ** 2)
                    loss += (self.fedprox_mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        if self.dp_enabled and self.privacy_engine is not None:
            try:
                current_epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
            except Exception:
                current_epsilon = 0.0

        avg_loss = total_loss / max(1, total_batches)
        return avg_loss, current_epsilon
