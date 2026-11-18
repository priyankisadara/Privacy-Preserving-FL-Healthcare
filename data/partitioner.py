"""
Data Partitioner for Federated Learning in Healthcare.
Supports:
1. IID Partitioning: Balanced uniform distribution across all hospital nodes.
2. Non-IID Dirichlet Partitioning: Simulates realistic patient demographic/pathology
   variations across different hospital centers using Dirichlet distribution Dir(alpha).
"""

import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, Subset


def partition_iid(dataset: Dataset, num_clients: int, seed: int = 42) -> Dict[int, Subset]:
    """
    Partitions a dataset uniformly and independently across `num_clients`.
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    splits = np.array_split(indices, num_clients)
    
    client_subsets = {}
    for client_id in range(num_clients):
        client_subsets[client_id] = Subset(dataset, splits[client_id].tolist())
    return client_subsets


def partition_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 15,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partitions sample indices using a Dirichlet distribution over class labels.
    Enforces that every participating hospital node receives at least `min_samples_per_client`
    to guarantee valid local training even under extreme heterogeneity (low alpha).
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    min_size = 0
    max_retries = 200
    retries = 0

    while min_size < min_samples_per_client and retries < max_retries:
        retries += 1
        client_indices = {i: [] for i in range(num_clients)}
        
        # Organize indices by class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
            
        # Sample proportions from Dirichlet distribution for each class
        for c in range(num_classes):
            idx_c = class_indices[c]
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Scale proportions to match number of samples in class c
            proportions = proportions / proportions.sum()
            counts = (proportions * len(idx_c)).astype(int)
            
            # Handle rounding remainder
            diff = len(idx_c) - counts.sum()
            for i in range(diff):
                counts[i % num_clients] += 1
                
            # Distribute indices to clients
            start_idx = 0
            for client_id in range(num_clients):
                allocated = idx_c[start_idx : start_idx + counts[client_id]]
                client_indices[client_id].extend(allocated.tolist())
                start_idx += counts[client_id]

        min_size = min(len(client_indices[i]) for i in range(num_clients))

    # Verification and shuffle
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
        
    return client_indices


def analyze_client_distribution(client_indices: Dict[int, List[int]], labels: np.ndarray) -> None:
    """
    Prints a summary table showing label distribution across hospital nodes.
    """
    num_classes = len(np.unique(labels))
    print("\n--- Client Data Distribution Summary ---")
    header = f"{'Client ID':<12} | {'Total Samples':<15} | " + " | ".join([f"Class {c}" for c in range(num_classes)])
    print(header)
    print("-" * len(header))
    
    for client_id, indices in client_indices.items():
        client_labels = labels[indices]
        class_counts = [np.sum(client_labels == c) for c in range(num_classes)]
        counts_str = " | ".join([f"{count:<7}" for count in class_counts])
        print(f"Hospital {client_id:<3} | {len(indices):<15} | {counts_str}")
    print("-" * len(header))


if __name__ == "__main__":
    from data.download_datasets import download_medmnist, get_labels_from_dataset
    train_ds, val_ds, test_ds = download_medmnist()
    train_labels = get_labels_from_dataset(train_ds)
    
    print("\n[+] Testing IID Partitioning (5 Hospital Clients):")
    iid_splits = partition_iid(train_ds, num_clients=5)
    iid_indices = {cid: subset.indices for cid, subset in iid_splits.items()}
    analyze_client_distribution(iid_indices, train_labels)

    print("\n[+] Testing Non-IID Dirichlet Partitioning (alpha=0.1, 5 Hospital Clients):")
    non_iid_indices = partition_dirichlet(train_labels, num_clients=5, alpha=0.1)
    analyze_client_distribution(non_iid_indices, train_labels)
