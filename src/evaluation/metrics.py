"""
Clinical Evaluation Metrics for Healthcare Model Performance.
Calculates Accuracy, AUROC, Macro F1, Precision, Recall, Loss,
and Optimal Classification Threshold (Youden's J Statistic).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from typing import Dict, Tuple


def evaluate_model(model: nn.Module, data_loader, device: torch.device) -> Dict[str, float]:
    """
    Evaluates PyTorch model on a DataLoader and computes comprehensive healthcare metrics.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Class 1 (Pneumonia)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    total_samples = len(all_targets)
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    try:
        auroc = roc_auc_score(all_targets, all_probs)
        # Compute Youden's J statistic: max(Sensitivity + Specificity - 1)
        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(thresholds[best_idx])
        # Metrics at optimal threshold
        opt_preds = (all_probs >= best_threshold).astype(int)
        opt_f1 = float(f1_score(all_targets, opt_preds, average="macro", zero_division=0))
        opt_accuracy = float(accuracy_score(all_targets, opt_preds))
    except Exception:
        auroc = 0.5
        best_threshold = 0.5
        opt_f1 = f1
        opt_accuracy = accuracy

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "optimal_threshold": float(best_threshold),
        "optimal_f1": float(opt_f1),
        "optimal_accuracy": float(opt_accuracy)
    }
