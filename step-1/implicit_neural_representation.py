import math
import random

import torch
import torch.nn as nn


class INR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_features=2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        sigma = 0.5
        x = [
            math.cos(2 * math.pi * random.gauss(0, sigma) * x[0]),
            math.sin(2 * math.pi * random.gauss(0, sigma) * x[1]),
        ]
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.tanh(self.out(x))
        print(x)
        return x


torch.manual_seed(41)
model = INR()
