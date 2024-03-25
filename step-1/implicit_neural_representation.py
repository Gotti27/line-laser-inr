from datetime import datetime
from random import uniform

import torch
from torch.utils.tensorboard import SummaryWriter

from inr_model import INR
from utils import *

UNIFORM_TRAINING_EPOCHS = 10
GRADIENT_BASED_TRAINING_EPOCHS = 0
INTRA_RAY_DEGREES = 1

torch.manual_seed(41)
model = INR()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    if epoch_index == 0:
        ray_image = np.zeros((500, 500, 1), np.uint8)

    for val_index in range(10000):  # 1000 batches
        x = np.random.uniform(0., 500., 128)
        y = np.random.uniform(0., 500., 128)

        inputs = [fall_to_nearest_ray([x[index], y[index]], [250, 250], INTRA_RAY_DEGREES) for index in range(128)]

        if epoch_index == 0:
            for p in inputs:
                if p[0] < 499 and p[1] < 499:
                    ray_image[round(p[1]), round(p[0])] = [255]
            cv.imshow('rays', ray_image)
            cv.waitKey(1)

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


def train_one_gradient_based_epoch(epoch_index, tb_writer, distr):
    running_loss = 0.
    last_loss = 0.

    for val_index in range(2000):  # 1000 batches
        sampled = np.random.choice(np.arange(len(distr)), size=128, p=distr, replace=False)
        _, cols, _ = gradient_image.shape
        biggs_x = (sampled // cols) + x_offset
        biggs_y = (sampled % cols) + y_offset

        inputs = [fall_to_nearest_ray([biggs_x[index], biggs_y[index]], [250, 250], INTRA_RAY_DEGREES) for index in
                  range(128)]

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
best_vloss = 1_000_000.

for epoch in range(UNIFORM_TRAINING_EPOCHS):
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
            val_x = np.random.uniform(0., 500., 128)
            val_y = np.random.uniform(0., 500., 128)

            v_inputs = [fall_to_nearest_ray([val_x[index], val_y[index]], [250, 250], INTRA_RAY_DEGREES) for index in
                        range(128)]
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
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

for epoch in range(GRADIENT_BASED_TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    gradient_image = np.zeros((500, 500, 1), np.float32)
    image = np.zeros((500, 500, 1), np.uint8)

    gradient_sum = 0.
    x_offset = uniform(0.0, 1.0)
    y_offset = uniform(0.0, 1.0)
    for i in range(500):
        for j in range(500):
            x = torch.tensor([[j + x_offset, i + y_offset]], dtype=torch.float32, requires_grad=True)
            output = model(x)
            output.backward()
            a, b = x.grad[0]
            value = math.sqrt((a ** 2) + (b ** 2))
            gradient_sum += value
            gradient_image[j, i] = value
            # image[i, j] = 255 if value > 1 else value

    print("sum ", gradient_sum)
    gradient_image /= gradient_sum
    flattened_distribution = gradient_image.flatten()

    image = gradient_image * 255.
    cv.imshow("gradient", image)
    cv.waitKey(1)

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_gradient_based_epoch(epoch_number, writer, flattened_distribution)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for val_v_index in range(100):
            sampled_index = np.random.choice(np.arange(len(flattened_distribution)), size=128, p=flattened_distribution)
            _, num_cols, _ = gradient_image.shape
            val_x = sampled_index // num_cols
            val_y = sampled_index % num_cols

            v_inputs = [fall_to_nearest_ray([val_x[index], val_y[index]], [250, 250], INTRA_RAY_DEGREES) for index in
                        range(128)]
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
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

print("done")
# cv.waitKey(0)
cv.destroyAllWindows()

model.train(False)

torch.save(model.state_dict(), 'model')
