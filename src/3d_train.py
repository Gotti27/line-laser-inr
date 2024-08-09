import argparse
import os
import time
from datetime import datetime

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dataset import INRPointsDataset, load_renders
from utils import *

target = 'bunny'
mode = 'gradient'  # 'gradient'
EPSILON = 0

if mode != 'uniform' and mode != 'gradient':
    raise RuntimeError("mode not valid")

UNIFORM_ITERATIONS = 10 if mode == 'uniform' else 0
UNIFORM_TRAINING_EPOCHS = 20

GRADIENT_ITERATIONS = 10 if mode == 'gradient' else 0
GRADIENT_BASED_TRAINING_EPOCHS = 20
# np.seterr(divide='ignore', invalid='ignore')


np.bool = np.bool_
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
pv.global_theme.allow_empty_mesh = True

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

if UNIFORM_ITERATIONS > 0:
    with open(f"history-{target}-uniform.txt", "a+") as history:
        history.write(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

if GRADIENT_ITERATIONS > 0:
    with open(f"history-{target}-gradient.txt", "a+") as history:
        history.write(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

mesh = pv.read(f'scenes/meshes/{target}.ply')
mesh = mesh.rotate_z(180)
mesh = mesh.rotate_x(90)
mesh = mesh.scale(10)
mesh.compute_normals(inplace=True)
if debug:
    p1 = pv.Plotter()
    p1.add_mesh(mesh, color='tan')
    p1.add_arrows(mesh.points, mesh.active_normals, color='black')
    p1.add_axes()
    p1.show_grid()
    p1.show()

image_folder = f'renders/{target}'
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images.sort(key=lambda name: int(name.split('_')[1]))

torch.manual_seed(41)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
model = INR3D(device=device)
model = model.to(device)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

load = True
if debug and load:
    model.load_state_dict(torch.load(f'3d-model-{target}-grad', map_location=device))

renders_matrices = load_renders(images, target)


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

    render = renders_matrices[image]['render'][:, :, 0:3]
    red_channel = render[:, :, 2] * 255
    _, red_channel = cv.threshold(red_channel, 100, 255, cv.THRESH_BINARY)

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))
    point_cloud_e = []
    point_cloud_u = []

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
    if debug:
        render = np.array(render)
        for p in points:
            cv.drawMarker(render, project_point(p[0].tolist(), R, t, K), [0, 255, 0], cv.MARKER_TILTED_CROSS, 1, 1)

        cv.imshow('foobar', render)
        cv.waitKey(0)
    '''

    for _ in range(laser_points):
        x, y, z = sample_point_from_plane([0, -2, 0], laser_norm)
        # FIXME: temporary condition to prevent central cluster on unknown points
        if x < -4 or x > 4 or z < -4 or z > 4:
            continue
        p = project_point([x, y, z], R, t, K)
        p_laser_center = project_point([laser_center[0], laser_center[1], laser_center[2]], R, t, K)

        '''
        direction = -laser_center[1] / (y - laser_center[1])
        far_point = (
            laser_center[0] + direction * (x - laser_center[0]), 0, laser_center[2] + direction * (z - laser_center[2]))

        p_far_point = np.array([far_point[0], far_point[1], far_point[2], 1.])
        p_far_point = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p_far_point
        p_far_point = [int(round(p_far_point[0, 0] / p_far_point[0, 2])),
                       int(round(p_far_point[0, 1] / p_far_point[0, 2]))]
        '''

        p_far_point = [int(round(i)) for i in np.array(p_laser_center) + 2 * (np.array(p) - np.array(p_laser_center))]

        line_points = [line_point for line_point in bresenham(p_far_point[0], p_far_point[1], p[0], p[1])]
        if side == 'right':
            line_points.reverse()

        unknown = True
        for point in line_points:
            if 0 < point[1] < 256 and 0 < point[0] < 256 and red_channel[point[1], point[0]] > 200:
                unknown = False
                break

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
            cv.drawMarker(render, p, [0, 0, 255], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)
        '''

        if unknown:
            points.append([[x, y, z], 0])
            point_cloud_u.append([x, y, z])
        else:
            points.append([[x, y, z], 1])
            point_cloud_e.append([x, y, z])

    '''
    if debug:
        mesh = pv.read('scenes/meshes/teapot.ply')
        mesh.rotate_x(90, inplace=True)
        mesh.rotate_y(180, inplace=True)
        point_cloud_u.extend(mesh.points)
        point_cloud_e.extend(mesh.points)
        point_cloud_e = pv.PolyData([[p_p * 10 for p_p in p] for p in point_cloud_e])
        point_cloud_u = pv.PolyData([[p_p * 10 for p_p in p] for p in point_cloud_u])
        point_cloud_e.plot(eye_dome_lighting=True)
        point_cloud_u.plot(eye_dome_lighting=True)
    '''

    return points


def create_uniform_dataset(silhouette_points=3000, laser_points=300):
    external = []
    internal = []
    unknown = []

    inputs = np.array([]).reshape(0, 3)

    for point in [[random.uniform(-4, 4), random.uniform(-4, 0), random.uniform(-4, 4)] for _ in
                  range(silhouette_points)]:
        label = silhouette_sampling(point)
        if label == 1:
            external.append(point)
        else:
            unknown.append(point)

    sampling_list = images.copy()
    for _ in range(len(sampling_list)):
        image = random.sample(sampling_list, 1)[0]
        sampling_list.remove(image)
        for point, label in laser_ray_sampling(image, laser_points):
            if label == 1:
                external.append(point)
            elif label == 0:
                unknown.append(point)
                if silhouette_sampling(point) == 1:
                    external.append(point)
                else:
                    unknown.append(point)
            else:
                internal.append(point)

    '''
    if debug:
        with open(f'dataset-{target}.pkl', 'wb') as f:
            pickle.dump(external, f)
            pickle.dump(unknown, f)
            pickle.dump(internal, f)
        
        with open(f'dataset-{target}.pkl', 'rb') as f:
            external = pickle.load(f)
            unknown = pickle.load(f)
            internal = pickle.load(f)
    '''
    print("Uniform raw dataset created - executing KNN")
    if debug:
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in unknown])
        point_cloud.plot(eye_dome_lighting=True, show_axes=True, show_bounds=True)
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in external])
        point_cloud.plot(eye_dome_lighting=True, show_axes=True, show_bounds=True)

    print(math.floor(math.sqrt(len(external) + len(internal) + len(unknown))))

    external, internal = pure_knn_point_classification(
        [[p_p * 10 for p_p in p] for p in external],
        [],
        [[p_p * 10 for p_p in p] for p in unknown],
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


def create_gradient_base_dataset(gradient_image_d, silhouette_points=3000, laser_points=300):
    external = []
    internal = []
    unknown = []

    inputs = np.array([]).reshape(0, 3)

    x = torch.linspace(-40, 40, 100, dtype=torch.float32, device='cpu')  # + offset
    y = torch.linspace(-40, 0, 50, dtype=torch.float32, device='cpu')  # + offset
    z = torch.linspace(-40, 40, 100, dtype=torch.float32, device='cpu')  # + offset
    X, Y, Z = torch.meshgrid(x, y, z)
    grid = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

    # for point in sample_from_gradient_image(gradient_image, silhouette_points):
    start = time.time()
    print("started silhouette sampling")
    for point in gibbs.gibbs_sampling_3d(gradient_image_d, silhouette_points, [0, 0, 0], grid):
        label = silhouette_sampling([p_p / 10 for p_p in point])
        if label == 1:
            external.append(point)
        else:
            unknown.append(point)
    print("finished silhouette sampling", time.time() - start)
    start = time.time()

    sampling_list = images.copy()
    for _ in range(360 * 2):
        image = random.sample(sampling_list, 1)[0]
        # FIXME: image = random.sample(sampling_list, 1)[0]
        # image = sampling_list[91 + 30]  # 90
        # print(image)
        for point, label in laser_ray_gradient_sampling(image, gradient_image_d, laser_points):
            if label == 1:
                external.append(point)
            elif label == 0:
                if silhouette_sampling([p_p / 10 for p_p in point]) == 1:
                    external.append(point)
                else:
                    unknown.append(point)
            else:
                internal.append(point)
        # print(f"{image} done")

    print("images done", time.time() - start)
    if debug:
        point_cloud = pv.PolyData(unknown)
        point_cloud.plot(eye_dome_lighting=True)
        point_cloud = pv.PolyData(external)
        point_cloud.plot(eye_dome_lighting=True)

    external, internal = pure_knn_point_classification(external, [], unknown, 5)

    if debug:
        point_cloud = pv.PolyData(internal)
        point_cloud.plot(eye_dome_lighting=True)
        point_cloud = pv.PolyData(external)
        point_cloud.plot(eye_dome_lighting=True)

    inputs = np.concatenate((inputs, external), axis=0)
    inputs = np.concatenate((inputs, internal), axis=0)

    labels = torch.tensor([[1] for _ in external] + [[0] for _ in internal], dtype=torch.float32,
                          requires_grad=True, device=device)

    dataset = [
        [torch.tensor(torch.from_numpy(inputs[i]), dtype=torch.float32, requires_grad=True, device=device), labels[i]]
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


for iteration in range(UNIFORM_ITERATIONS):
    print(f"iteration: {iteration}")
    uniform_dataset = INRPointsDataset(create_uniform_dataset(100000, 100))

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

    for epoch in range(UNIFORM_TRAINING_EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch_uniformly(epoch_number, writer)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            '''
            eval_err = rmse_model_evaluation(model, mesh.vertices, mesh.vertex_normals)
            print('LOSS train {} valid {}'.format(avg_loss, eval_err))  # abs_model_evaluation(model, mesh.points)
            '''
            print('LOSS train {} valid {}'.format(avg_loss, 'unknown'))

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

    abs_error = abs_model_evaluation(model, mesh.points)
    mae_error = mae_model_evaluation(model, mesh.points, mesh.active_normals)
    rmse_error = rmse_model_evaluation(model, mesh.points, mesh.active_normals)
    print(f"EVAL ERR: {abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}")
    with open(f"history-{target}-uniform.txt", "a+") as history:
        history.write(f"{abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}\n")


def compute_gradient_image_from_model():
    # offset = random.uniform(0, 80 / 100)
    x = torch.linspace(-40, 40, 100, dtype=torch.float32, device=device, requires_grad=True)
    y = torch.linspace(-40, 0, 50, dtype=torch.float32, device=device, requires_grad=True)
    z = torch.linspace(-40, 40, 100, dtype=torch.float32, device=device, requires_grad=True)

    X, Y, Z = torch.meshgrid(x, y, z)

    points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)
    output = model(points)
    grad_outputs = torch.autograd.grad(outputs=output, inputs=points, grad_outputs=torch.ones_like(output),
                                       is_grads_batched=False)[0]

    grad_outputs = grad_outputs.detach().cpu()
    gradient = torch.tensor([math.sqrt((x ** 2) + (y ** 2) + (z ** 2)) for [x, y, z] in grad_outputs],
                            dtype=torch.float32, device='cpu')

    return gradient.to(device), output


def laser_ray_gradient_sampling(image, gradient_image, laser_points=300):
    points = []
    side = image.split('_')[2]

    K = renders_matrices[image]['K']
    R = renders_matrices[image]['R']
    t = renders_matrices[image]['t']
    laser_center = renders_matrices[image]['laser_center']
    laser_norm = renders_matrices[image]['laser_norm']

    a, b, c = laser_norm
    d = -(a * laser_center[0] + b * laser_center[1] + c * laser_center[2])

    render = renders_matrices[image]['render'][:, :, 0:3]
    red_channel = render[:, :, 2] * 255
    _, red_channel = cv.threshold(red_channel, 100, 255, cv.THRESH_BINARY)

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))
    point_cloud_e = []
    point_cloud_u = []

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

        points.append([np.squeeze(
            np.asarray(find_plane_line_intersection([a, b, c, d], camera_position, np.array(laser_point_world)))
        ) * 10, -1])

    sampled_points, grid_points = sample_point_from_plane_gradient([0, -20, 0], laser_norm, model, laser_points)
    sampled_points /= 10
    grid_points /= 10
    '''
    plt.figure()
    plt.imshow(gradient_image[:, :, 50])
    plt.show(block=False)
    

    render_copy = copy.deepcopy(render)
    for point in points:  # .flatten(start_dim=0, end_dim=1):
        p = project_point((point[0] / 10).tolist(), R, t, K)
        print(p)
        cv.drawMarker(render_copy, p, [255, 255, 0], cv.MARKER_TILTED_CROSS, 2, 2)
    cv.imshow("test-1", render_copy)
    cv.waitKey(1)
    '''

    dbg = False
    if dbg:
        for point in sampled_points:  # .flatten(start_dim=0, end_dim=1):
            p = project_point(point.numpy().tolist(), R, t, K)
            cv.drawMarker(render, p, [255, 255, 0], cv.MARKER_DIAMOND, 2, 1)
        cv.imshow("test", render)
        cv.waitKey(1)

    # sampled_points.cpu().detach().numpy()

    for [x, y, z] in sampled_points:  # .flatten(start_dim=0, end_dim=1):
        # print(x, y, z)
        p = project_point([x, y, z], R, t, K)
        p_laser_center = project_point([laser_center[0], laser_center[1], laser_center[2]], R, t, K)

        '''
        direction = -laser_center[1] / (y - laser_center[1])
        far_point = (
            laser_center[0] + direction * (x - laser_center[0]), 0, laser_center[2] + direction * (z - laser_center[2]))

        p_far_point = np.array([far_point[0], far_point[1], far_point[2], 1.])
        p_far_point = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p_far_point
        p_far_point = [int(round(p_far_point[0, 0] / p_far_point[0, 2])),
                       int(round(p_far_point[0, 1] / p_far_point[0, 2]))]
        '''

        p_far_point = [int(round(i)) for i in np.array(p_laser_center) + 2 * (np.array(p) - np.array(p_laser_center))]

        line_points = [line_point for line_point in bresenham(p_far_point[0], p_far_point[1], p[0], p[1])]
        if side == 'right':
            line_points.reverse()

        unknown = True
        for point in line_points:
            if 0 < point[1] < 256 and 0 < point[0] < 256 and red_channel[point[1], point[0]] > 200:
                unknown = False
                break

        if not unknown:
            for point in line_points:
                if 0 < point[1] < 256 and 0 < point[0] < 256:
                    render[point[1], point[0]] = [0, 255, 0]
                    if red_channel[point[1], point[0]] > 200:
                        break

        '''
        if debug:
            cv.drawMarker(render, p_far_point, [255, 255, 0], cv2.MARKER_DIAMOND, 2, 1)
            cv.drawMarker(render, p, [0, 0, 255], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)
        '''

        if unknown:
            points.append([[x * 10, y * 10, z * 10], 0])
            point_cloud_u.append([x * 10, y * 10, z * 10])
        else:
            points.append([[x * 10, y * 10, z * 10], 1])
            point_cloud_e.append([x * 10, y * 10, z * 10])

    '''
    if debug:
        mesh = pv.read('scenes/meshes/teapot.ply')
        mesh.rotate_x(90, inplace=True)
        mesh.rotate_y(180, inplace=True)
        point_cloud_u.extend(mesh.points)
        point_cloud_e.extend(mesh.points)
        point_cloud_e = pv.PolyData([[p_p * 10 for p_p in p] for p in point_cloud_e])
        point_cloud_u = pv.PolyData([[p_p * 10 for p_p in p] for p in point_cloud_u])
        point_cloud_e.plot(eye_dome_lighting=True)
        point_cloud_u.plot(eye_dome_lighting=True)
    '''

    return points


def train_one_epoch_gradient(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    training_loader = torch.utils.data.DataLoader(gradient_based_dataset, batch_size=64, shuffle=True,
                                                  generator=torch.Generator(device=device), num_workers=0)

    for batch_index, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = (outputs + 1) / 2
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TODO: print(f"batch {batch_index} done, loss: {loss.item()}", end='\r')

        if batch_index == len(training_loader) - 1:
            last_loss = running_loss / len(training_loader)  # loss per batch
            print('  batch {} loss: {}'.format(batch_index + 1, last_loss))
            tb_x = epoch_index * 100 + batch_index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


model.train(False)
torch.save(model.state_dict(), f'3d-model-{target}')
print("Uniform training completed, intermediate model saved")
model.train(True)

for iteration in range(GRADIENT_ITERATIONS):
    print(f"iteration: {iteration}")
    gradient_image, model_output_grid = compute_gradient_image_from_model()
    gradient_image = gradient_image.view(100, 50, 100)
    model_output_grid = model_output_grid.view(100, 50, 100)
    print("gradient image done")

    gradient_image = gradient_image.to('cpu').detach().numpy()
    model_output_grid = model_output_grid.to('cpu').detach().numpy()

    if debug:
        plane = gradient_image[:, 25, :]
        fig = plt.figure()
        plt.imshow(plane)
        plt.show()

        plane = model_output_grid[:, 25, :]
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

    gradient_based_dataset = INRPointsDataset(create_gradient_base_dataset(gradient_image, 100000, 100))

    for epoch in range(GRADIENT_BASED_TRAINING_EPOCHS):
        print('EPOCH - gradient based - {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch_gradient(epoch_number, writer)

        model.eval()
        with torch.no_grad():
            print('LOSS train {} valid {}'.format(avg_loss, 'unknown'))

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

    abs_error = abs_model_evaluation(model, mesh.points)
    mae_error = mae_model_evaluation(model, mesh.points, mesh.active_normals)
    rmse_error = rmse_model_evaluation(model, mesh.points, mesh.active_normals)
    print(f"EVAL ERR: {abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}")
    with open(f"history-{target}-gradient.txt", "a+") as history:
        history.write(f"{abs_error} {mae_error[0]} {rmse_error[0]} {rmse_error[2]}\n")

print(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

if UNIFORM_ITERATIONS > 0:
    with open(f"history-{target}-uniform.txt", "a+") as history:
        history.write(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

if GRADIENT_ITERATIONS > 0:
    with open(f"history-{target}-gradient.txt", "a+") as history:
        history.write(f"done {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}\n")

model.train(False)
torch.save(model.state_dict(), f'3d-model-{target}')

if debug:
    cv.waitKey(0)
    cv.destroyAllWindows()
