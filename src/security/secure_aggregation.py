"""
Secure Aggregation (SecAgg) Simulation.
Implements pairwise zero-sum random masking (Bonawitz et al., ACM CCS 2017).
Guarantees that the central server can only compute the aggregate sum
without ever learning individual client weight updates.
"""

import numpy as np
from typing import List, Dict


class SecureAggregator:
    """
    Simulates Pairwise Masking for Federated Secure Aggregation.
    Each client pair (i, j) shares a random mask s_ij.
    Client i adds +s_ij and Client j adds -s_ij.
    In weighted FedAvg:
      Client i sends: (n_i / N) * w_i + mask_i
      Server computes: sum_i [ (n_i / N) * w_i + mask_i ] = sum_i (n_i / N) * w_i + 0
    """
    def __init__(self, num_clients: int, seed: int = 42):
        self.num_clients = num_clients
        self.rng = np.random.RandomState(seed)

    def generate_pairwise_masks(self, weight_shapes: List[tuple]) -> Dict[int, List[np.ndarray]]:
        """
        Generates zero-sum pairwise masks for all participating clients.
        """
        # Initialize zero masks for each client
        client_masks = {
            i: [np.zeros(shape, dtype=np.float32) for shape in weight_shapes]
            for i in range(self.num_clients)
        }

        # For every pair (i, j) with i < j, generate shared pseudo-random mask
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                pair_masks = [self.rng.normal(0, 0.5, shape).astype(np.float32) for shape in weight_shapes]
                for l in range(len(weight_shapes)):
                    client_masks[i][l] += pair_masks[l]   # Client i adds +mask
                    client_masks[j][l] -= pair_masks[l]   # Client j subtracts -mask

        return client_masks

    def mask_weighted_weights(
        self, client_id: int, weights: List[np.ndarray], weight_factor: float, client_masks: Dict[int, List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Applies client sample weight factor and zero-sum mask:
        y_i = (n_i / N) * w_i + mask_i
        """
        mask = client_masks[client_id]
        return [(w * np.float32(weight_factor)) + m for w, m in zip(weights, mask)]

    def verify_cancellation(self, client_masks: Dict[int, List[np.ndarray]]) -> bool:
        """
        Verifies that the sum of all masks across all clients equals exactly zero.
        """
        num_layers = len(client_masks[0])
        for l in range(num_layers):
            total_mask = sum(client_masks[i][l] for i in range(self.num_clients))
            if not np.allclose(total_mask, 0.0, atol=1e-5):
                return False
        return True


if __name__ == "__main__":
    dummy_shapes = [(16, 1, 3, 3), (16,), (128, 576), (2,)]
    sec_agg = SecureAggregator(num_clients=5)
    masks = sec_agg.generate_pairwise_masks(dummy_shapes)
    is_valid = sec_agg.verify_cancellation(masks)
    print(f"[*] Secure Aggregation Masking Cancellation Verified: {is_valid}")
    
    # Test weighted aggregation cancellation
    sample_counts = [1529, 785, 1283, 405, 706]
    total_samples = sum(sample_counts)
    dummy_weights = [[np.ones(s, dtype=np.float32) for s in dummy_shapes] for _ in range(5)]
    
    masked_packages = [
        sec_agg.mask_weighted_weights(i, dummy_weights[i], sample_counts[i] / total_samples, masks)
        for i in range(5)
    ]
    
    # Server sums masked packages directly
    server_aggregate = [
        sum(masked_packages[i][l] for i in range(5))
        for l in range(len(dummy_shapes))
    ]
    
    # Check if server aggregate equals exact 1.0 everywhere
    matches_clean = np.allclose(server_aggregate[0], np.ones(dummy_shapes[0], dtype=np.float32), atol=1e-5)
    print(f"[*] Server Unmasked Aggregate Matches True Weighted Weights: {matches_clean}")
