import math
from datetime import datetime

import numpy as np
import rff
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def gear(angle):
    return 30 + (5 * math.sin(10 * math.radians(angle)))


def oracle(point):
    radius, angle = convert_cartesian_to_polar((50, 50), point)
    diff = radius - gear(angle)
    if -2 < diff < 2:
        return 0
    return np.sign(diff)


def convert_cartesian_to_polar(center, point):
    vector = (point[0] - center[0], point[1] - center[1])
    radius = np.linalg.norm(vector)
    angle = math.atan2(vector[1], vector[0])

    angle_degrees = math.degrees(angle)
    if angle_degrees < 0:
        angle_degrees += 360

    return radius, angle_degrees


def convert_polar_to_cartesian(angle, radius, center):
    center_x, center_y = center
    return (radius * math.cos(math.radians(angle))) + center_x, (radius * math.sin(math.radians(angle))) + center_y


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


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for val_index in range(1000):  # 1000 batches
        x = np.random.choice(range(100), size=128, replace=True)
        y = np.random.choice(range(100), size=128, replace=True)

        inputs = [[x[index], y[index]] for index in range(128)]

        labels = torch.tensor([[oracle(p)] for p in inputs], dtype=torch.float32, requires_grad=True)

        optimizer.zero_grad()
        outputs = model(torch.tensor(inputs, dtype=torch.float32, requires_grad=True))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if val_index % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(val_index + 1, last_loss))
            tb_x = epoch_index * 100 + val_index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('step-1-runs/model_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 10
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for val_v_index in range(100):
            val_x = np.random.choice(range(100), size=128, replace=True)
            val_y = np.random.choice(range(100), size=128, replace=True)

            v_inputs = [[val_x[index], val_y[index]] for index in range(128)]
            v_labels = torch.tensor([[oracle(p)] for p in v_inputs], dtype=torch.float32, requires_grad=True)

            v_outputs = model(torch.tensor(v_inputs, dtype=torch.float32, requires_grad=True))
            vloss = loss_fn(v_outputs, v_labels)
            running_vloss += vloss

    avg_vloss = running_vloss / (val_v_index + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

model.train(False)

torch.save(model.state_dict(), 'model')
