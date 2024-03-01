from datetime import datetime
from random import uniform

import torch
from torch.utils.tensorboard import SummaryWriter

from inr_model import INR
from utils import *

torch.manual_seed(41)
model = INR()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for val_index in range(1000):  # 1000 batches
        x = np.random.uniform(0., 100., 128)
        y = np.random.uniform(0., 100., 128)

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


def train_one_gibbs_epoch(epoch_index, tb_writer, distr):
    running_loss = 0.
    last_loss = 0.

    for val_index in range(1000):  # 1000 batches
        sampled = np.random.choice(np.arange(len(distr)), size=128, p=distr)
        _, cols, _ = gradient_image.shape
        x = sampled // cols
        y = sampled % cols

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

UNIFORM_TRAINING_EPOCHS = 40
GIBBS_TRAINING_EPOCHS = 20
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
            val_x = np.random.uniform(0., 100., 128)
            val_y = np.random.uniform(0., 100., 128)

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
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

for epoch in range(GIBBS_TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    gradient_image = np.zeros((100, 100, 1), np.float32)
    image = np.zeros((100, 100, 1), np.uint8)

    gradient_sum = 0.
    offset = uniform(0.0, 1.0)
    print(offset)
    for i in range(100):
        for j in range(100):
            x = torch.tensor([[i + offset, j + offset]], dtype=torch.float32, requires_grad=True)
            output = model(x)
            output.backward()
            a, b = x.grad[0]
            value = math.sqrt(a ** 2 + b ** 2)
            gradient_sum += value
            gradient_image[i, j] = value  # 255 if value > 1 else value

    print("sum ", gradient_sum)
    gradient_image /= gradient_sum
    flattened_distribution = gradient_image.flatten()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_gibbs_epoch(epoch_number, writer, flattened_distribution)

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
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

print("done")
# cv.waitKey(0)
# cv.destroyAllWindows()

model.train(False)

torch.save(model.state_dict(), 'model')
