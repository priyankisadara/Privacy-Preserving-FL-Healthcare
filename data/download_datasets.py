"""
Dataset Downloader and Loader for Healthcare Benchmarks.
Downloads MedMNIST (PneumoniaMNIST) and applies medical data augmentations.
"""

import os
import medmnist
from medmnist import INFO
import numpy as np
import torch
from torchvision import transforms


def download_medmnist(dataset_name: str = "pneumoniamnist", data_dir: str = "./data/raw"):
    """
    Downloads and prepares MedMNIST dataset with clinical data augmentations for training.
    """
    os.makedirs(data_dir, exist_ok=True)
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])

    # 1. Training Augmentations (flips, slight rotations, affine jitter)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Validation & Test: Deterministic normalization only
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print(f"[*] Downloading / Loading '{dataset_name}' into {data_dir}...")
    train_dataset = DataClass(split="train", transform=train_transform, download=True, root=data_dir)
    val_dataset = DataClass(split="val", transform=eval_transform, download=True, root=data_dir)
    test_dataset = DataClass(split="test", transform=eval_transform, download=True, root=data_dir)

    print(f"[+] Loaded successfully:")
    print(f"    - Train samples: {len(train_dataset)}")
    print(f"    - Validation samples: {len(val_dataset)}")
    print(f"    - Test samples: {len(test_dataset)}")
    print(f"    - Number of classes: {len(info['label'])} ({info['label']})")
    
    return train_dataset, val_dataset, test_dataset


def get_labels_from_dataset(dataset) -> np.ndarray:
    """Extracts raw labels as a 1D NumPy array for partitioning."""
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels).flatten()
    elif hasattr(dataset, "targets"):
        return np.array(dataset.targets).flatten()
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
        return np.array(labels).flatten()


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Computes inverse frequency class weights to balance Cross-Entropy Loss:
    w_c = total_samples / (num_classes * count_c)
    """
    classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    num_classes = len(classes)
    weights = total_samples / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    train_ds, val_ds, test_ds = download_medmnist()
    labels = get_labels_from_dataset(train_ds)
    weights = compute_class_weights(labels)
    print(f"[*] Computed Balanced Class Weights: Normal (0) = {weights[0]:.3f}, Pneumonia (1) = {weights[1]:.3f}")
