import argparse
import random
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from inr_model import INR3D
from utils import *

UNIFORM_TRAINING_EPOCHS = 8
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

# import point cloud as tensor
with open("data/ball.xyz") as point_cloud_file:
    point_cloud = np.array(
        [[float(n) for n in line.rstrip().split(" ")] for line in point_cloud_file])  # if random.random() >= 0

# set up the model
torch.manual_seed(41)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
model = INR3D(device=device)
model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for batch_index in range(UNIFORM_BATCH_NUMBER):
        external = []
        internal = []
        unknown = []

        inputs = np.array([]).reshape(0, 3)
        for z in range(5, 70):
            center = [0, 0]
            point_level = list(filter(lambda p: round(p[2]) == z, point_cloud))
            level_points = random.sample(point_level, 100 if len(point_level) > 100 else len(point_level))

            external.extend(level_points)

            for second in level_points:
                radius, angle = convert_cartesian_to_polar(center=center, point=second)

                e_x, e_y = convert_polar_to_cartesian(angle, radius + random.randint(0, 30), center)
                i_x, i_y = convert_polar_to_cartesian(angle, random.randint(0, round(radius)), center)
                external.append([e_x, e_y, z])
                internal.append([i_x, i_y, z])
            # print(external)
            # print(internal)
            '''
            e, inner, u = generate_laser_points(start_point, angle)
            external.extend(random.sample(e, 40 if len(e) > 40 else len(e)))
            internal.extend(random.sample(inner, 10 if len(inner) > 10 else len(inner)))
            unknown.extend(random.sample(u, 40 if len(u) > 40 else len(u)))
            '''

        # external, internal = knn_point_classification(external, internal, unknown, 5)

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


for epoch in range(UNIFORM_TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    model.eval()

    '''
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
    '''
    print('LOSS train {} valid {}'.format(avg_loss, 'unknown'))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    '''
    if avg_vloss < best_validation_loss:
        best_validation_loss = avg_vloss
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    '''

    epoch_number += 1

print(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
model.train(False)
torch.save(model.state_dict(), '3d-model')

if debug:
    cv.waitKey(0)
    cv.destroyAllWindows()
