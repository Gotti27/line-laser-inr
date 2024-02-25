import cv2 as cv
import numpy as np
import rff
import torch
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


torch.manual_seed(41)
model = INR()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('model'))
image = np.zeros((100, 100, 1), np.uint8)
zoom = np.zeros((100, 100, 1), np.uint8)

for i in range(100):
    for j in range(100):
        output = model(torch.tensor([[i / 2, (j + 100) / 2], [i, j]])).detach()

        zoom[i, j] = (output[0, 0] + 1) * 127.5
        image[i, j] = (output[1, 0] + 1) * 127.5

        cv.imshow("full", image)
        cv.imshow("zoom", zoom)
        cv.waitKey(1)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('full.png', image)
cv.imwrite('zoom.png', zoom)
