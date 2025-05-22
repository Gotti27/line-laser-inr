import argparse
import os
from datetime import datetime

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dataset import INRPointsDataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--target", dest="target", help="Target object", default=None)
parser.add_argument("-d", "--debug", dest="debug", help="Enable debug mode", action="store_true", default=False)
args = parser.parse_args()

target = args.target if args.target is not None else 'Dragon'
num_workers = int(os.cpu_count())
TRAINING_EPOCHS = 10000
NUMBER_OF_POINTS = 2 ** 15

np.bool = np.bool_
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
pv.global_theme.allow_empty_mesh = True

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/model_trainer_{}'.format(timestamp))
epoch_number = 0
best_validation_loss = 1_000_000.

debug = args.debug
if debug:
    print("---DEBUG MODE ACTIVATED---")

print(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
with open(f"history-{target}.txt", "a+") as history:
    history.write(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

# Load mesh and rescaling in - 0.5, 0.5 bounding box
target = 'igea'
mesh = pv.read(f'scenes/meshes/{target}.ply')  # pv.examples.download_dragon()
# mesh.compute_normals(inplace=True)
mesh = mesh.translate([-dim for dim in mesh.center])
x_length = mesh.bounds[1] - mesh.bounds[0]
y_length = mesh.bounds[3] - mesh.bounds[2]
z_length = mesh.bounds[5] - mesh.bounds[4]

max_length = max(x_length, y_length, z_length)
scaling_factor = 1 / max_length

mesh = mesh.scale([scaling_factor, scaling_factor, scaling_factor])
mesh.save(f'scenes/meshes/{target}-small.ply')
print(mesh.bounds)

if debug:
    p1 = pv.Plotter()
    p1.add_mesh(mesh, color='tan')
    # p1.add_bounding_box([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], color='red')
    # p1.add_arrows(mesh.points, mesh.active_normals, color='black')
    p1.add_axes()
    p1.show_grid()
    p1.show()

torch.manual_seed(41)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
model = INR3D(device=device)
model = model.to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5 * (10 ** -4))  # lr=0.001)  #


def lr_lambda(step):
    return 0.1 ** (step / 5000)


# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.01 ** (1 / 5000))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def create_dataset():
    m = trimesh.Trimesh(mesh.points, faces=mesh.faces.reshape((mesh.n_cells, 4))[:, 1:], validate=True, process=True,
                        use_embree=True)
    if not m.is_watertight:
        raise RuntimeError('not watertight')

    points = torch.rand((NUMBER_OF_POINTS, 3), dtype=torch.float32, device='cpu') - 0.5
    print("sampled")

    labels = [m.contains(chunk.cpu().numpy()) for chunk in points.chunk(100)]
    labels = np.concatenate(labels)
    # labels = pool.map(lambda p: m.contains(p.cpu().numpy()), points.chunk(num_workers))
    # labels = np.array(labels)

    print("contains executed")

    if debug:
        p1 = pv.Plotter()
        p1.add_points(np.array([e for i, e in enumerate(points) if labels[i]]), color='red')
        # p1.add_points(np.array([e for i, e in enumerate(points) if not p[i]]), color='blue')
        # p1.add_mesh(mesh, color='tan')
        # p1.add_bounding_box([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], color='red')
        # p1.add_arrows(mesh.points, mesh.active_normals, color='black')
        p1.add_axes()
        p1.show_grid()
        p1.show()

    points = points.cuda()
    labels = torch.tensor([[0 if l else 1] for l in labels], dtype=torch.float32, requires_grad=True, device=device)

    # torch.from_numpy(points[i]).type(torch.float32).requires_grad_(True).to(device)
    dataset = [
        [points[i], labels[i]]
        for i
        in range(len(points))]
    print("dataset created")
    return dataset


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for batch_index, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = (outputs + 1) / 2
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_index == len(training_loader) - 1:
            last_loss = running_loss / len(training_loader)
            print('  batch {} loss: {}'.format(batch_index + 1, last_loss))
            tb_x = epoch_index * 100 + batch_index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


dataset = None
training_loader = None

for epoch in range(TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    if epoch % 1 == 0:
        print(f"creating epoch {epoch} dataset")
        dataset = INRPointsDataset(create_dataset())

        training_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                                      generator=torch.Generator(device=device), num_workers=0)

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    scheduler.step()
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        print(f'LOSS train {avg_loss} valid -- Learning Rate: {scheduler.get_last_lr()[0]}')

    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss},
                       epoch_number + 1)
    writer.flush()

    epoch_number += 1

    '''
    abs_error = abs_model_evaluation(model, mesh.points)
    mae_error = mae_model_evaluation(model, mesh.points, mesh.active_normals)
    rmse_error = rmse_model_evaluation(model, mesh.points, mesh.active_normals)
    print(f"EVAL ERR: {abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}")
    with open(f"history-{target}.txt", "a+") as history:
        history.write(f"{abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}\n")
    '''

    if epoch_number % 1000 == 0:
        # scheduler.step()
        model.train(False)
        torch.save(model.state_dict(), f'3d-model-{target}-{epoch_number}')

print(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
with open(f"history-{target}.txt", "a+") as history:
    history.write(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

model.train(False)
torch.save(model.state_dict(), f'3d-model-{target}')

if debug:
    cv.waitKey(0)
    cv.destroyAllWindows()
