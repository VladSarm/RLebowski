"""
Policy network for REINFORCE on Bowling.
CNN (2 convolutional layers) + MLP (2 fully connected layers).
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Input:  (batch, 1, 75, 160) ROI channel
    Output: (batch, 6)           logits for 6 actions
    """

    def __init__(self):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 5, kernel_size=2, stride=1, padding=0)
        self.relu = nn.ReLU()

        # MLP
        self.fc1 = nn.Linear(15200, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, 1, 75, 160)"""
        x = self.relu(self.conv1(x))  # -> (batch, 16, 39, 81)
        x = self.relu(self.conv2(x))  # -> (batch, 5, 38, 80)
        x = x.reshape(x.size(0), -1)  # -> (batch, 15200)
        x = self.relu(self.fc1(x))    # -> (batch, 512)
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
