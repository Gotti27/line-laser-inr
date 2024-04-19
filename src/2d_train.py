import argparse
import random
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from inr_model import INR2D
from utils import *

UNIFORM_TRAINING_EPOCHS = 15
GRADIENT_BASED_TRAINING_EPOCHS = 5
INTRA_RAY_DEGREES = 1
UNIFORM_BATCH_NUMBER = 50
GRADIENT_BASED_BATCH_NUMBER = 10

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/model_trainer_{}'.format(timestamp))
epoch_number = 0
best_validation_loss = 1_000_000.

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", dest="debug", help="Enable debug mode", action="store_true", default=False)
args = parser.parse_args()

debug = args.debug
if debug:
    print("---DEBUG MODE ACTIVATED---")

print(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

# set up the model
torch.manual_seed(41)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
model = INR2D(device=device)
model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for batch_index in range(UNIFORM_BATCH_NUMBER):
        if debug:
            laser_rays = np.zeros((500, 500, 1), np.uint8)
            laser_points = np.zeros((500, 500, 1), np.uint8)
            laser_points.fill(255)
            laser_points_after_knn = np.zeros((500, 500, 1), np.uint8)
            laser_points_after_knn.fill(255)
        else:
            laser_rays = None
            laser_points = None
            laser_points_after_knn = None

        external = []
        internal = []
        unknown = []

        inputs = np.array([]).reshape(0, 2)
        for _ in range(400):
            start_point = [250, 250]  # [random.randint(0, 500), random.randint(0, 500)]
            angle = random.uniform(0., 360.)

            if debug:
                simulate_laser_ray(start_point, angle, 1, laser_rays)
            e, inner, u = generate_laser_points(start_point, angle)
            external.extend(random.sample(e, 40 if len(e) > 40 else len(e)))
            internal.extend(random.sample(inner, 10 if len(inner) > 10 else len(inner)))
            unknown.extend(random.sample(u, 40 if len(u) > 40 else len(u)))

        external, internal = knn_point_classification(external, internal, unknown, 5)

        if debug:
            for e in list(filter(lambda p: 0 < p[0] < 500 and 0 < p[1] < 500, external)):
                laser_points_after_knn[e[1], e[0]] = 150
                # laser_points[e[1], e[0]] = 0

            for inne in list(filter(lambda p: 0 < p[0] < 500 and 0 < p[1] < 500, internal)):
                laser_points[inne[1], inne[0]] = 0
                laser_points_after_knn[inne[1], inne[0]] = 0

            cv.imshow("laser negative points", laser_points)
            cv.imshow("laser negative points after knn", laser_points_after_knn)
            cv.imshow("laser rays", laser_rays)
            cv.waitKey(1)

        inputs = np.concatenate((inputs, external), axis=0)
        inputs = np.concatenate((inputs, internal), axis=0)

        labels = torch.tensor([[1] for _ in external] + [[-1] for _ in internal], dtype=torch.float32,
                              requires_grad=True, device=device)

        optimizer.zero_grad()
        outputs = model(torch.tensor(np.array(inputs), dtype=torch.float32, requires_grad=True, device=device))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"batch {batch_index} done, loss: {loss.item()}", end='\r')

        if batch_index == UNIFORM_BATCH_NUMBER - 1:
            last_loss = running_loss / UNIFORM_BATCH_NUMBER  # loss per batch
            print('  batch {} loss: {}'.format(batch_index + 1, last_loss))
            tb_x = epoch_index * 100 + batch_index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def gradient_based_sampling(points, size):
    t = []
    result = []
    for point in points:
        point_tensor = torch.tensor([[point[0], point[1]]],
                                    dtype=torch.float32, requires_grad=True, device=device)
        point_output = model(point_tensor)
        point_output.backward()
        t.append(point_tensor.grad[0])

    gradient_values = np.array([math.sqrt((point[0] ** 2) + (point[1] ** 2)) for point in t])
    gradient_values /= sum(gradient_values)
    distribution = gradient_values.flatten()

    for sampled in np.random.choice(np.array([d for d in range(len(distribution))]),
                                    size=(size if len(t) > size else len(t)),
                                    p=distribution, replace=False):
        result.append(points[sampled])
    return result


def train_one_gradient_based_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for batch_index in range(GRADIENT_BASED_BATCH_NUMBER):
        if debug:
            laser_points = np.zeros((500, 500, 1), np.uint8)
        else:
            laser_points = None

        external = []
        internal = []
        unknown = []

        inputs = np.array([]).reshape(0, 2)
        for _ in range(200):
            start_point = [250, 250]
            angle = random.uniform(0., 360.)

            e, inner, u = generate_laser_points(start_point, angle)

            external.extend(gradient_based_sampling(e, 15))
            internal.extend(gradient_based_sampling(inner, 10))
            unknown.extend(gradient_based_sampling(u, 15))

        external, internal = knn_point_classification(external, internal, unknown, 5)

        inputs = np.concatenate((inputs, external), axis=0)
        inputs = np.concatenate((inputs, internal), axis=0)

        labels = torch.tensor([[1] for _ in external] + [[-1] for _ in internal], dtype=torch.float32,
                              requires_grad=True, device=device)

        if debug:
            for p in inputs:
                px = round(p[1])
                py = round(p[0])
                if 0 <= px < 500 and 0 <= py < 500:
                    laser_points[px, py] = 255

            cv.imshow('gradient based sampling', laser_points)
            cv.imwrite(f'extracted_{epoch_index}.png', laser_points)
            cv.waitKey(1)

        optimizer.zero_grad()
        outputs = model(torch.tensor(inputs, dtype=torch.float32, requires_grad=True, device=device))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"batch {batch_index} done, loss: {loss.item()}", end='\r')

        running_loss += loss.item()
        if batch_index == GRADIENT_BASED_BATCH_NUMBER - 1:
            last_loss = running_loss / GRADIENT_BASED_BATCH_NUMBER  # loss per batch
            print('  batch {} loss: {}'.format(batch_index + 1, last_loss))
            tb_x = epoch_index * 100 + batch_index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


for epoch in range(UNIFORM_TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    model.eval()

    with torch.no_grad():
        for val_index in range(100):
            val_x = np.random.uniform(0., 500., 128)
            val_y = np.random.uniform(0., 500., 128)

            v_inputs = [[val_x[i], val_y[i]] for i in range(128)]
            v_labels = torch.tensor([[realistic_oracle(p)] for p in v_inputs], dtype=torch.float32, requires_grad=True,
                                    device=device)

            v_outputs = model(torch.tensor(v_inputs, dtype=torch.float32, requires_grad=True, device=device))
            vloss = loss_fn(v_outputs, v_labels)
            running_vloss += vloss

    avg_vloss = running_vloss / (val_index + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_validation_loss:
        best_validation_loss = avg_vloss
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

for epoch in range(GRADIENT_BASED_TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    if debug:
        gradient_image = np.zeros((500, 500, 1), np.float32)
        image = np.zeros((500, 500, 1), np.uint8)

        gradient_sum = 0.
        for i in range(500):
            for j in range(500):
                gradient_tensor = torch.tensor([[j, i]], dtype=torch.float32, requires_grad=True, device=device)
                output = model(gradient_tensor)
                output.backward()
                gradient_x, gradient_y = gradient_tensor.grad[0]
                value = math.sqrt((gradient_x ** 2) + (gradient_y ** 2))
                gradient_sum += value
                gradient_image[j, i] = value

        gradient_image /= gradient_sum
        flattened_distribution = gradient_image.flatten()

        image = gradient_image * 255.
        cv.imshow("gradient", image)
        cv.imwrite(f"images/gradient_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{epoch_number}.png", image)
        cv.waitKey(1)

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_gradient_based_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for val_v_index in range(100):
            val_x = np.random.uniform(0., 500., 128)
            val_y = np.random.uniform(0., 500., 128)

            v_inputs = [[val_x[index], val_y[index]] for index in range(128)]
            v_labels = torch.tensor([[realistic_oracle(p)] for p in v_inputs], dtype=torch.float32, requires_grad=True,
                                    device=device)

            v_outputs = model(torch.tensor(v_inputs, dtype=torch.float32, requires_grad=True, device=device))
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
    if avg_vloss < best_validation_loss:
        gradient_tensor = avg_vloss
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

print(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
model.train(False)
torch.save(model.state_dict(), 'model')

if debug:
    cv.waitKey(0)
    cv.destroyAllWindows()
