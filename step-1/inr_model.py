import rff
import torch.nn as nn


class INR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sigma = 0.5
        # self.B = torch.tensor([[random.gauss(0, sigma), random.gauss(0, sigma)] for _ in range(64)])
        self.encoding = rff.layers.GaussianEncoding(sigma=sigma, input_size=2, encoded_size=64)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.encoding(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.tanh(self.out(x))

        return x
