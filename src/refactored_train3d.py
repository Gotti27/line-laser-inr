import argparse
import copy
import os
from datetime import datetime

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dataset import INRPointsDataset, load_renders
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--target", dest="target", help="Target object", default=None)
parser.add_argument("-i", "--images", dest="images_number", help="Number of input images", type=int, default=0)
parser.add_argument("-m", "--mode", dest="mode", help="mode: uniform or gradient", choices=['uniform', 'gradient'],
                    default='uniform')
parser.add_argument("-e", "--epsilon", dest="epsilon", help="epsilon to smooth gradient distribution", type=float,
                    default=0)
parser.add_argument("-d", "--debug", dest="debug", help="Enable debug mode", action="store_true", default=False)
args = parser.parse_args()

target = args.target if args.target is not None else 'Dragon-small'
mode = args.mode
NUMBER_IMAGES = args.images_number
EPSILON = args.epsilon

CORES_FRACTION = 1
num_workers = int(os.cpu_count() * CORES_FRACTION)

if mode != 'uniform' and mode != 'gradient':
    raise RuntimeError("mode not valid")

UNIFORM_ITERATIONS = 10 if mode == 'uniform' else 0
UNIFORM_TRAINING_EPOCHS = 200

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

if UNIFORM_ITERATIONS > 0:
    with open(f"history-{target}-uniform.txt", "a+") as history:
        history.write(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} IMAGES: {NUMBER_IMAGES}\n")

mesh = pv.read(f'scenes/meshes/{target}.ply')
mesh = mesh.rotate_z(180)

mesh.compute_normals(inplace=True)
if debug:
    p1 = pv.Plotter()
    p1.add_mesh(mesh, color='tan')
    # p1.add_arrows(mesh.points, mesh.active_normals, color='black')
    p1.add_axes()
    p1.show_grid()
    p1.show()

image_folder = f'renders/{target}'
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images = random.sample(images, NUMBER_IMAGES if 0 < NUMBER_IMAGES < len(images) else len(images))
images.sort(key=lambda name: int(name.split('_')[1]))

torch.manual_seed(41)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
model = INR3D(device=device)
model = model.to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                   gamma=0.01 ** (1 / 100))  # gamma=0.01 # gamma=0.01 ** (1 / 5000)

load = False
if debug and load:
    model.load_state_dict(torch.load(f'models/3d-model-{target}-gradient', map_location=device))

renders_matrices = load_renders(images, target)


def efficient_silhouette_sampling(points: np.ndarray):
    labels = np.full(points.shape[0], np.False_)

    for image in images:
        render_depth = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)

        K = renders_matrices[image]['K']
        R = renders_matrices[image]['R']
        t = renders_matrices[image]['t']

        p = project_points(points, R, t, K).T

        img_height, img_width = 1024, 1024

        valid_mask = (0 <= p[0]) & (p[0] < img_width) & \
                     (0 <= p[1]) & (p[1] < img_height)
        valid_indices = np.where(valid_mask)[1]

        valid_points = p[:, valid_indices].T
        # valid_points = p[:, np.squeeze(np.asarray(valid_mask))].T

        valid_points = np.squeeze(np.asarray(valid_points))

        temp_depth = copy.deepcopy(render_depth[:, :, 3])
        depth_mask = render_depth[valid_points[:, 1], valid_points[:, 0], 3] == 0
        render_depth[valid_points[:, 1], valid_points[:, 0]] = np.array([0, 255, 0, 0])
        # render_depth[valid_points[:, 1], valid_points[:, 0], ~depth_mask] = np.array([0, 0, 255, 0])
        render_depth[:, :, 3] = temp_depth

        ###

        temp_labels = np.full((points.shape[0]), np.False_, dtype=np.bool_)
        temp_labels[valid_indices] = temp_depth[valid_points[:, 1], valid_points[:, 0]] == 0.0

        ###
        labels |= temp_labels

        cv.imshow("Projected Points", render_depth)
        cv.waitKey(1)

    cv.destroyAllWindows()
    return labels


def silhouette_sampling(point):
    x, y, z = point
    # for image in list(filter(lambda img: 'right' in img, images)):
    for image in images:
        K = renders_matrices[image]['K']
        R = renders_matrices[image]['R']
        t = renders_matrices[image]['t']
        render_depth = renders_matrices[image]['render']

        p = project_point([x, y, z], R, t, K)
        depth = render_depth[:, :, 3]

        is_outside = p[0] < 0 or p[0] >= 256 or p[1] < 0 or p[1] >= 256
        if not is_outside and depth[p[1], p[0]] == 0:
            return 1
    return -1


def laser_ray_sampling(image, laser_points):
    points = []
    degree = int(image.split('_')[1])
    side = image.split('_')[2]

    K = renders_matrices[image]['K']
    R = renders_matrices[image]['R']
    t = renders_matrices[image]['t']
    laser_center = renders_matrices[image]['laser_center']
    laser_norm = renders_matrices[image]['laser_norm']

    a, b, c = laser_norm
    d = -(a * laser_center[0] + b * laser_center[1] + c * laser_center[2])

    render = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)[:, :, 0:3]
    red_channel = render[:, :, 2] * 255
    _, red_channel = cv.threshold(red_channel, 100, 255, cv.THRESH_BINARY)

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))
    point_cloud_e = []
    point_cloud_u = []

    '''
    for [u, v] in np.column_stack(np.where(red_channel > 0)):
        laser_point_camera = np.array(
            [v - (red_channel.shape[1] / 2), u - (red_channel.shape[0] / 2), K[0][0], 1])
        laser_point_world = np.concatenate([
            np.concatenate([R.T, np.array(- R.T @ t).reshape(3, 1)], axis=1),
            np.array([[0, 0, 0, 1]])
        ], axis=0) @ laser_point_camera

        laser_point_world = [laser_point_world[0] / laser_point_world[3],
                             laser_point_world[1] / laser_point_world[3],
                             laser_point_world[2] / laser_point_world[3]]

        world_point = np.squeeze(
            np.asarray(find_plane_line_intersection([a, b, c, d], camera_position, np.array(laser_point_world)))
        )

        points.append([world_point, -1])
    '''

    '''
    if debug:
        render = np.array(render)
        for p in points:
            cv.drawMarker(render, project_point(p[0].tolist(), R, t, K), [0, 255, 0], cv.MARKER_TILTED_CROSS, 1, 1)

        cv.imshow('foobar', render)
        cv.waitKey(0)
    '''

    p_laser_center = project_point([laser_center[0], laser_center[1], laser_center[2]], R, t, K)
    p_laser_center = np.array(p_laser_center)
    points = sample_points_from_plane([0, 0, 0], laser_norm, laser_points).T
    projected_points_camera = project_points(points, R, t, K)
    dataset = []

    projected_points_camera = np.asarray(projected_points_camera)
    points = np.asarray(points)
    for index in range(points.shape[0]):
        original_point = points[index]
        p = projected_points_camera[index]

        '''
        direction = -laser_center[1] / (y - laser_center[1])
        far_point = (
            laser_center[0] + direction * (x - laser_center[0]), 0, laser_center[2] + direction * (z - laser_center[2]))

        p_far_point = np.array([far_point[0], far_point[1], far_point[2], 1.])
        p_far_point = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p_far_point
        p_far_point = [int(round(p_far_point[0, 0] / p_far_point[0, 2])),
                       int(round(p_far_point[0, 1] / p_far_point[0, 2]))]
        '''

        p_far_point = [int(round(i)) for i in p_laser_center + 2 * (p - p_laser_center)]

        line_points = [line_point for line_point in bresenham(p_far_point[0], p_far_point[1], p[0], p[1])]
        if side == 'right':
            line_points.reverse()

        unknown = True
        for point in line_points:
            if 0 < point[1] < 1024 and 0 < point[0] < 1024:
                render[point[1], point[0], 1] = 255

            if 0 < point[1] < 1024 and 0 < point[0] < 1024 and red_channel[point[1], point[0]] > 200:
                unknown = False
                break

        '''
        if not unknown:
            for point in line_points:
                if 0 < point[1] < 256 and 0 < point[0] < 256:
                    render[point[1], point[0]] = [0, 255, 0]
                    if red_channel[point[1], point[0]] > 200:
                        break

        '''
        if debug:
            render = np.array(render)
            cv.drawMarker(render, p_far_point, [255, 255, 0], cv.MARKER_DIAMOND, 2, 1)
            cv.drawMarker(render, p, [255, 0, 0], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)

        if unknown:
            dataset.append([original_point, 0])
            point_cloud_u.append(original_point)
        else:
            dataset.append([original_point, 1])
            point_cloud_e.append(original_point)

    pl = pv.Plotter()
    pl.add_mesh(mesh)
    # pl.add_points(pv.PolyData(np.array(point_cloud_u)), color='red')
    pl.add_points(pv.PolyData(np.array(point_cloud_e)), color='green')
    pl.enable_eye_dome_lighting()
    pl.show_grid()
    pl.show()

    return dataset


def create_uniform_dataset(silhouette_points=3000, laser_points=300):
    external = []
    internal = []
    unknown = []

    inputs = np.array([]).reshape(0, 3)

    '''
    pool = multiprocess.Pool(num_workers)
    labels = pool.map(lambda p: [p, silhouette_sampling(p)],
                      [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)] for _ in
                       range(silhouette_points)])
    '''

    points = [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)] for _ in
              range(silhouette_points)]

    points = np.array(points)

    labels = efficient_silhouette_sampling(points)
    '''

    points = np.array(points)
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_points(points[labels], color='green')
    pl.add_points(points[~labels], color='red')
    pl.enable_eye_dome_lighting()
    pl.show_grid()
    pl.show_axes()
    pl.show()
    ####

    '''
    external = points[labels]
    unknown = points[~labels]

    '''
    sampling_list = images.copy()
    to_check = []
    for _ in range(len(sampling_list)):
        image = random.sample(sampling_list, 1)[0]
        sampling_list.remove(image)
        for point, label in laser_ray_sampling(image, laser_points):
            if label == 1:
                external.append(point)
            elif label == 0:
                to_check.append(point)
            else:
                pass
                # internal.append(point)

    labels = efficient_silhouette_sampling(to_check)

    external.extend(to_check[labels].tolist())
    unknown.extend(to_check[~labels].tolist())
    '''

    print("Uniform raw dataset created - executing KNN")
    if debug:
        p1 = pv.Plotter()
        p1.add_mesh(mesh, color='tan')
        p1.add_points(external)
        p1.add_axes()
        p1.show_grid()
        p1.show()

        '''
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p[0]] for p in unknown_l])
        point_cloud.plot(eye_dome_lighting=True, show_axes=True, show_bounds=True)
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p[0]] for p in external_l])
        point_cloud.plot(eye_dome_lighting=True, show_axes=True, show_bounds=True)
        '''

    # print(math.floor(math.sqrt(len(external) + len(internal) + len(unknown))))

    flag = False
    if flag:
        return pure_knn_point_classification(
            external.tolist(),
            [],
            unknown.tolist(),
            5
        )

    external, internal = pure_knn_point_classification(
        external.tolist(),  # [[p_p * 10 for p_p in p] for p in external],
        [],
        unknown.tolist(),  # [[p_p * 10 for p_p in p] for p in unknown],
        5
    )

    print("Uniform dataset created")

    if debug:
        p1 = pv.Plotter()
        p1.add_mesh(mesh, color='tan')
        p1.add_points(pv.PolyData(internal))
        p1.add_axes()
        p1.show_grid()
        p1.show()

        p1 = pv.Plotter()
        p1.add_mesh(mesh, color='tan')
        p1.add_points(pv.PolyData(external))
        p1.add_axes()
        p1.show_grid()
        p1.show()

    # evaluate_point_classification(mesh, external, internal)

    inputs = np.concatenate((inputs, external), axis=0)
    inputs = np.concatenate((inputs, internal), axis=0)

    labels = torch.tensor([[1] for _ in external] + [[0] for _ in internal], dtype=torch.float32,
                          requires_grad=True, device=device)

    print(f"Total number of points in the dataset {len(inputs)}")
    dataset = [
        [torch.from_numpy(inputs[i]).type(torch.float32).requires_grad_(True).to(device), labels[i]]
        for i
        in range(len(inputs))]
    return dataset


def train_one_epoch_uniformly(epoch_index, tb_writer):
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


for epoch in range(UNIFORM_TRAINING_EPOCHS):
    print(f"iteration: {epoch}")
    uniform_dataset = INRPointsDataset(create_uniform_dataset(2 ** 15, 0))  # (((2 ** 15)) // len(images))

    '''
    uniform_dataset_train, uniform_dataset_val = torch.utils.data.random_split(uniform_dataset, [.8, .2],
                                                                               generator=torch.Generator(device=device))
    '''
    uniform_dataset_train = uniform_dataset

    training_loader = torch.utils.data.DataLoader(uniform_dataset_train, batch_size=64, shuffle=True,
                                                  generator=torch.Generator(device=device), num_workers=0)
    '''
    validation_loader = torch.utils.data.DataLoader(uniform_dataset_val, batch_size=64, shuffle=True,
                                                    generator=torch.Generator(device=device), num_workers=0)
    '''

    # for epoch in range(UNIFORM_TRAINING_EPOCHS):
    # print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch_uniformly(epoch_number, writer)

    scheduler.step()
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        '''
        eval_err = rmse_model_evaluation(model, mesh.vertices, mesh.vertex_normals)
        print('LOSS train {} valid {}'.format(avg_loss, eval_err))  # abs_model_evaluation(model, mesh.points)
        '''
        print(f'LOSS train {avg_loss} valid -- Learning Rate: {scheduler.get_last_lr()[0]}')

    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss},
                       epoch_number + 1)
    writer.flush()

    '''
    if avg_vloss < best_validation_loss:
        best_validation_loss = avg_vloss
        model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    '''
    epoch_number += 1

    # abs_error = abs_model_evaluation(model, mesh.points)
    # mae_error = mae_model_evaluation(model, mesh.points, mesh.active_normals)
    # rmse_error = rmse_model_evaluation(model, mesh.points, mesh.active_normals)
    # print(f"EVAL ERR: {abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}")
    # with open(f"history-{target}-uniform.txt", "a+") as history:
    #    history.write(f"{abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}\n")

print(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

if UNIFORM_ITERATIONS > 0:
    with open(f"history-{target}-uniform.txt", "a+") as history:
        history.write(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

model.train(False)
torch.save(model.state_dict(), f'3d-model-{target}-{mode}')

if debug:
    cv.waitKey(0)
    cv.destroyAllWindows()
