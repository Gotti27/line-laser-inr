import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import multiprocess
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
UNIFORM_TRAINING_EPOCHS = 60
GRADIENT_TRAINING_EPOCHS = 60

np.bool = np.bool_
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
pv.global_theme.allow_empty_mesh = True

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

training_directory = f"models/{timestamp}_{target}_{mode}"
Path(training_directory).mkdir(parents=True, exist_ok=True)
print("Directory created")

writer = SummaryWriter(f'{training_directory}/model_trainer_{timestamp}')
epoch_number = 0
best_validation_loss = 1_000_000.

debug = args.debug
if debug:
    print("---DEBUG MODE ACTIVATED---")

print(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

with open(f"{training_directory}/history-{target}.txt", "a+") as history:
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
                                                   gamma=0.01 ** (1 / 25))  # gamma=0.01 # gamma=0.01 ** (1 / 5000)

'''
'''
load = True
if debug and load:
    model.load_state_dict(
        torch.load('models/20250425_011458_Dragon-small/3d-model-Dragon-small-60', map_location=device))

renders_matrices = load_renders(images, target, debug)


def efficient_silhouette_sampling(points: np.ndarray):
    def sample_from_image(image):
        if debug:
            render_depth = cv.imread(renders_matrices[image]['render'], cv.IMREAD_UNCHANGED)
        else:
            render_depth = renders_matrices[image]['render']

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
        render_depth[valid_points[:, 1], valid_points[:, 0]] = np.array([0, 255, 0, 0])
        # render_depth[valid_points[:, 1], valid_points[:, 0], ~depth_mask] = np.array([0, 0, 255, 0])
        render_depth[:, :, 3] = temp_depth

        ###

        temp_labels = np.full((points.shape[0]), np.False_, dtype=np.bool_)
        temp_labels[valid_indices] = temp_depth[valid_points[:, 1], valid_points[:, 0]] == 0.0

        return temp_labels

    pool = multiprocess.Pool(num_workers)
    result = pool.map(sample_from_image, images)

    labels = np.logical_or.reduce(result)

    return labels


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
    p_laser_center = project_point([laser_center[0], laser_center[1], laser_center[2]], R, t, K)

    sampled_points = sample_points_from_plane([0, 0, 0], laser_norm, laser_points).T

    projected_sampled_points = project_points(sampled_points, R, t, K)

    for sampled_point, projected_point in zip(np.squeeze(np.asarray(sampled_points)),
                                              np.squeeze(np.asarray(projected_sampled_points))):

        p_far_point = [int(round(i)) for i in
                       np.array(p_laser_center) + 2 * (projected_point - np.array(p_laser_center))]

        line_points = [line_point for line_point in
                       bresenham(p_far_point[0], p_far_point[1], projected_point[0], projected_point[1])]
        if side == 'right':
            line_points.reverse()

        unknown = True
        for point in line_points:
            if 0 < point[1] < red_channel.shape[0] and 0 < point[0] < red_channel.shape[1] \
                    and red_channel[point[1], point[0]] > 200:
                unknown = False
                break

        '''
        if debug:
            if not unknown:
                for point in line_points:
                    if 0 < point[1] < 1024 and 0 < point[0] < 1024:
                        render[point[1], point[0]] = [0, 255, 0]
                        if red_channel[point[1], point[0]] > 200:
                            break

            render = np.array(render)
            cv.drawMarker(render, p_far_point, [255, 255, 0], cv.MARKER_DIAMOND, 2, 1)
            cv.drawMarker(render, projected_point, [255, 0, 0], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)
        '''
        if unknown:
            points.append([sampled_point, 0])
        else:
            points.append([sampled_point, 1])

    '''
    if debug:
        depth = renders_matrices[image]['render'][:, :, 3]
        for i in range(len(render)):
            for j in range(len(render)):
                if depth[i][j] == 0 and np.array_equal(render[i][j], [0, 0, 0]):
                    render[i][j] = [1, 255, 255]

        cv.imshow('foobar', render)
        cv.waitKey(0)
    '''

    return points


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
    external_laser = []
    unknown_laser = []

    sampling_list = images.copy()
    to_check = []
    for _ in range(len(sampling_list)):
        image = random.sample(sampling_list, 1)[0]
        sampling_list.remove(image)
        for point, label in laser_ray_sampling(image, laser_points):
            if label == 1:
                external_laser.append(point)
            elif label == 0:
                to_check.append(point)
            else:
                pass
                # internal.append(point)

    to_check = np.array(to_check)
    labels = efficient_silhouette_sampling(to_check)
    external_laser = np.concatenate([np.array(external_laser), to_check[labels]])

    external = np.concatenate([external, np.array(external_laser)])
    unknown = np.concatenate([unknown, to_check[~labels]])
    # unknown = np.concatenate([unknown, np.array(unknown_laser)])
    # labels = efficient_silhouette_sampling(to_check)

    # external.extend(to_check[labels].tolist())
    # unknown.extend(to_check[~labels].tolist())

    '''
    external_laser = np.array(external_laser)
    unknown_laser = np.array(unknown_laser)

    external_laser = external_laser[
        np.random.choice(external_laser.shape[0], replace=False, size=silhouette_points)]

    external = np.concatenate([external, np.array(external_laser)])
    # unknown = np.concatenate([unknown, np.array(unknown_laser)])
    # labels = efficient_silhouette_sampling(to_check)

    # external.extend(to_check[labels].tolist())
    # unknown.extend(to_check[~labels].tolist())
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


if mode == 'uniform':
    for epoch in range(UNIFORM_TRAINING_EPOCHS):
        print(f"iteration: {epoch}")
        uniform_dataset = INRPointsDataset(create_uniform_dataset(200000, 200))  # (((2 ** 15)) // len(images))
        # 200000

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

        with open(f"{training_directory}/history-{target}.txt", "a+") as history:
            history.write(f"LOSS train {avg_loss} valid -- Learning Rate: {scheduler.get_last_lr()[0]}\n")

        if epoch_number % 10 == 0:
            # scheduler.step()
            model.train(False)
            torch.save(model.state_dict(), f'{training_directory}/3d-model-{target}-{epoch_number}')


def compute_gradient_image_from_model():
    x = torch.linspace(-0.5, 0.5, 100, dtype=torch.float32, device=device, requires_grad=True) + offset_x
    y = torch.linspace(-0.5, 0.5, 100, dtype=torch.float32, device=device, requires_grad=True) + offset_y
    z = torch.linspace(-0.5, 0.5, 100, dtype=torch.float32, device=device, requires_grad=True) + offset_z

    X, Y, Z = torch.meshgrid(x, y, z)

    points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)
    output = model(points)
    grad_outputs = torch.autograd.grad(outputs=output, inputs=points, grad_outputs=torch.ones_like(output),
                                       is_grads_batched=False)[0]

    grad_outputs = grad_outputs.detach()
    gradient = torch.sqrt((grad_outputs ** 2).sum(dim=1))

    return gradient, output


def laser_ray_gradient_sampling(image, gradient_image_d, laser_points):
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
    p_laser_center = project_point([laser_center[0], laser_center[1], laser_center[2]], R, t, K)

    sampled_points, grid_points = sample_point_from_plane_gradient([0, 0, 0], laser_norm, model, laser_points)
    projected_sampled_points = project_points(sampled_points, R, t, K)

    for sampled_point, projected_point in zip(np.squeeze(np.asarray(sampled_points)),
                                              np.squeeze(np.asarray(projected_sampled_points))):

        p_far_point = [int(round(i)) for i in
                       np.array(p_laser_center) + 2 * (projected_point - np.array(p_laser_center))]

        line_points = [line_point for line_point in
                       bresenham(p_far_point[0], p_far_point[1], projected_point[0], projected_point[1])]
        if side == 'right':
            line_points.reverse()

        unknown = True
        for point in line_points:
            if 0 < point[1] < red_channel.shape[0] and 0 < point[0] < red_channel.shape[1] \
                    and red_channel[point[1], point[0]] > 200:
                unknown = False
                break

        '''
        if debug:
            if not unknown:
                for point in line_points:
                    if 0 < point[1] < 1024 and 0 < point[0] < 1024:
                        render[point[1], point[0]] = [0, 255, 0]
                        if red_channel[point[1], point[0]] > 200:
                            break

            render = np.array(render)
            cv.drawMarker(render, p_far_point, [255, 255, 0], cv.MARKER_DIAMOND, 2, 1)
            cv.drawMarker(render, projected_point, [255, 0, 0], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)
        '''
        if unknown:
            points.append([sampled_point, 0])
        else:
            points.append([sampled_point, 1])

    '''
    if debug:
        depth = renders_matrices[image]['render'][:, :, 3]
        for i in range(len(render)):
            for j in range(len(render)):
                if depth[i][j] == 0 and np.array_equal(render[i][j], [0, 0, 0]):
                    render[i][j] = [1, 255, 255]

        cv.imshow('foobar', render)
        cv.waitKey(0)
    '''

    return points


def create_gradient_dataset(gradient_image_d, silhouette_points=3000, laser_points=300):
    external = []
    internal = []
    unknown = []

    inputs = np.array([]).reshape(0, 3)

    x = torch.linspace(-0.5, 0.5, 100, dtype=torch.float32, device=device) + offset_x
    y = torch.linspace(-0.5, 0.5, 100, dtype=torch.float32, device=device) + offset_y
    z = torch.linspace(-0.5, 0.5, 100, dtype=torch.float32, device=device) + offset_z
    X, Y, Z = torch.meshgrid(x, y, z)
    grid = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

    points = gibbs.gibbs_sampling_3d(gradient_image_d, silhouette_points, [0, 0, 0], grid)
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
    external_laser = []
    unknown_laser = []

    sampling_list = images.copy()
    to_check = []
    for _ in range(len(sampling_list)):
        image = random.sample(sampling_list, 1)[0]
        sampling_list.remove(image)
        for point, label in laser_ray_gradient_sampling(image, gradient_image_d, laser_points):
            if label == 1:
                external_laser.append(point)
            elif label == 0:
                to_check.append(point)
            else:
                pass
                # internal.append(point)

    if debug:
        pl = pv.Plotter()
        pl.add_mesh(mesh)
        pl.add_points(np.array(to_check), color='orange')
        pl.add_points(np.array(external_laser))
        pl.enable_eye_dome_lighting()
        pl.show_grid()
        pl.show()

    to_check = np.array(to_check)
    labels = efficient_silhouette_sampling(to_check)
    external_laser = np.concatenate([np.array(external_laser), to_check[labels]])

    external = np.concatenate([external, np.array(external_laser)])
    unknown = np.concatenate([unknown, to_check[~labels]])
    # unknown = np.concatenate([unknown, np.array(unknown_laser)])
    # labels = efficient_silhouette_sampling(to_check)

    # external.extend(to_check[labels].tolist())
    # unknown.extend(to_check[~labels].tolist())

    '''
    external_laser = np.array(external_laser)
    unknown_laser = np.array(unknown_laser)

    external_laser = external_laser[
        np.random.choice(external_laser.shape[0], replace=False, size=silhouette_points)]

    external = np.concatenate([external, np.array(external_laser)])
    # unknown = np.concatenate([unknown, np.array(unknown_laser)])
    # labels = efficient_silhouette_sampling(to_check)

    # external.extend(to_check[labels].tolist())
    # unknown.extend(to_check[~labels].tolist())
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


def train_one_epoch_gradient(epoch_index, tb_writer):
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


if mode == 'gradient':
    for epoch in range(GRADIENT_TRAINING_EPOCHS):
        print(f"iteration: {epoch}")

        offset_x = random.uniform(-0.5 / 100, 0.5 / 100)
        offset_y = random.uniform(-0.5 / 100, 0.5 / 100)
        offset_z = random.uniform(-0.5 / 100, 0.5 / 100)

        gradient_image, model_output_grid = compute_gradient_image_from_model()
        gradient_image = gradient_image.view(100, 100, 100)
        model_output_grid = model_output_grid.view(100, 100, 100)
        print("gradient image done")

        gradient_image = gradient_image.to('cpu').detach().numpy()
        model_output_grid = model_output_grid.to('cpu').detach().numpy()

        if debug:
            plane = gradient_image[:, 50, :]
            fig = plt.figure()
            plt.imshow(plane)
            plt.show()

            plane = model_output_grid[:, 50, :]
            fig = plt.figure()
            plt.imshow(plane)
            plt.show(block=True)

            plane = gradient_image[50, :, :]
            fig = plt.figure()
            plt.imshow(plane)
            plt.show(block=True)

            plane = gradient_image[:, :, 50]
            fig = plt.figure()
            plt.imshow(plane)
            plt.show(block=True)

            plane = model_output_grid[:, :, 50]
            fig = plt.figure()
            plt.imshow(plane)
            plt.show(block=True)

        gradient_image += EPSILON

        gradient_dataset = INRPointsDataset(
            create_gradient_dataset(gradient_image, 200000, 200))  # (((2 ** 15)) // len(images))
        # 200000

        '''
        gradient_dataset_train, gradient_dataset_val = torch.utils.data.random_split(gradient_dataset, [.8, .2],
                                                                                   generator=torch.Generator(device=device))
        '''
        gradient_dataset_train = gradient_dataset

        training_loader = torch.utils.data.DataLoader(gradient_dataset_train, batch_size=64, shuffle=True,
                                                      generator=torch.Generator(device=device), num_workers=0)
        '''
        validation_loader = torch.utils.data.DataLoader(gradient_dataset_val, batch_size=64, shuffle=True,
                                                        generator=torch.Generator(device=device), num_workers=0)
        '''

        # for epoch in range(GRADIENT_TRAINING_EPOCHS):
        # print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch_gradient(epoch_number, writer)

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
        # with open(f"history-{target}-gradient.txt", "a+") as history:
        #    history.write(f"{abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}\n")

        with open(f"{training_directory}/history-{target}.txt", "a+") as history:
            history.write(f"LOSS train {avg_loss} valid -- Learning Rate: {scheduler.get_last_lr()[0]}\n")

        if epoch_number % 10 == 0:
            # scheduler.step()
            model.train(False)
            torch.save(model.state_dict(), f'{training_directory}/3d-model-{target}-{epoch_number}')

print(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

with open(f"{training_directory}/history-{target}.txt", "a+") as history:
    history.write(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

model.train(False)
torch.save(model.state_dict(), f'{training_directory}/3d-model-{target}-{mode}')

if debug:
    cv.waitKey(0)
    cv.destroyAllWindows()
