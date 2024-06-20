import numpy as np
import torch
import torch.nn as nn


class INR2D(nn.Module):
    def __init__(self, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        sigma = 0.5 / 10
        self.B = torch.normal(0, sigma, size=(64, 2)).to(device)
        # self.encoding = rff.layers.GaussianEncoding(sigma=sigma, input_size=2, encoded_size=64)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)

    def gaussian_encoding(self, x):
        vp = 2 * np.pi * x @ self.B.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

    def forward(self, x):
        x = self.gaussian_encoding(x)  # self.encoding(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.tanh(self.out(x))

        return x


class INR3D(nn.Module):
    def __init__(self, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        sigma = 0.5 / 10
        self.B = torch.normal(0, sigma, size=(128, 3)).to(device)

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)

    def gaussian_encoding(self, x):
        vp = 2 * np.pi * x @ self.B.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

    def forward(self, x):
        x = self.gaussian_encoding(x)  # self.encoding(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.tanh(self.out(x))

        return x


def sign_loss(output, target):
    return sum([1 for i in range(len(output)) if (output[i] * target[i]) < 0])
