import time

import cv2 as cv
import numpy as np
import torch

from inr_model import INR
from marching_squares import marching_squares

torch.manual_seed(41)
model = INR()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('model'))
image = np.zeros((500, 500, 1), np.uint8)
zoom = np.zeros((500, 500, 1), np.uint8)

for i in range(500):
    for j in range(500):
        output = model(torch.tensor([[i / 4 + 200, j / 4 + 250], [i, j]])).detach()

        zoom[i, j] = (output[0, 0] + 1) * 127.5
        image[i, j] = (output[1, 0] + 1) * 127.5

cv.imshow('full', image)
cv.imshow('zoom', zoom)
cv.waitKey(1)

for t in reversed(range(5, 50, 5)):
    image_copy = image.copy()
    zoom_copy = zoom.copy()
    marching_squares(image_copy, t, 200)
    marching_squares(zoom_copy, t, 200)
    cv.imshow('full', image_copy)
    cv.imshow('zoom', zoom_copy)
    cv.waitKey(1)
    time.sleep(5)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('full.png', image)
cv.imwrite('zoom.png', zoom)
