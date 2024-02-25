import cv2 as cv
import numpy as np
import torch

from inr_model import INR

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
