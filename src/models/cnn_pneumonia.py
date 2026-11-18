"""
Medical Image Classification Model (Pneumonia Detection).
Optimized CNN for 28x28 Chest X-Rays with high representational capacity.
Uses GroupNorm for full compatibility with Differential Privacy (Opacus DP-SGD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PneumoniaCNN(nn.Module):
    """
    Enhanced CNN architecture for Pneumonia diagnosis from Chest X-Rays.
    3 Convolutional Blocks (32 -> 64 -> 128) with Group Normalization and Dropout.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(PneumoniaCNN, self).__init__()
        
        # Block 1: Conv -> GroupNorm -> LeakyReLU -> MaxPool (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=32)
        
        # Block 2: Conv -> GroupNorm -> LeakyReLU -> MaxPool (14x14 -> 7x7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
        
        # Block 3: Conv -> GroupNorm -> LeakyReLU -> MaxPool (7x7 -> 3x3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(num_groups=16, num_channels=128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classifier Head: 128 * 3 * 3 = 1152 -> 128 -> 2
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.leaky_relu(self.norm1(self.conv1(x)), negative_slope=0.1))
        # Block 2
        x = self.pool(F.leaky_relu(self.norm2(self.conv2(x)), negative_slope=0.1))
        # Block 3
        x = self.pool(F.leaky_relu(self.norm3(self.conv3(x)), negative_slope=0.1))
        
        # Flatten & Dense
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    return PneumoniaCNN(in_channels=in_channels, num_classes=num_classes)
