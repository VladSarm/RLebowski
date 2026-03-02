"""
Policy network for REINFORCE on Bowling.
CNN (2 convolutional layers) + MLP (2 fully connected layers).
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Input:  (batch, 1, 84, 84)  grayscale
    Output: (batch, 6)           logits for 6 actions
    """

    def __init__(self):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.relu = nn.ReLU()

        # MLP
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, 1, 84, 84)"""
        x = self.relu(self.conv1(x))  # → (batch, 16, 20, 20)
        x = self.relu(self.conv2(x))  # → (batch, 32, 9, 9)
        x = x.reshape(x.size(0), -1)  # → (batch, 2592)
        x = self.relu(self.fc1(x))    # → (batch, 256)
        x = self.fc2(x)               # → (batch, 6)
        return x

    def get_action_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax over logits to get action probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_action_log_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Log-softmax for numerical stability."""
        logits = self.forward(x)
        return torch.log_softmax(logits, dim=-1)
