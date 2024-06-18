import argparse
import os
import pickle
from datetime import datetime

import cv2
import pyvista as pv
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from inr_model import INR3D
from utils import *

UNIFORM_TRAINING_EPOCHS = 100  # 150  # 50  # 50  # 100  # 350  # 200  # 6
GRADIENT_BASED_TRAINING_EPOCHS = 5  # 1  # 2

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

image_folder = 'renders'
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images.sort(key=lambda name: int(name.split('_')[1]))

# set up the model
torch.manual_seed(41)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
model = INR3D(device=device)
model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

load = False
if load:
    model.load_state_dict(torch.load('3d-model', map_location=device))


def load_render():
    renders = {}

    for image in list(filter(lambda img: 'right' in img, images)):
        degree = image.split('_')[1]
        side = image.split('_')[2]
        with open(f'renders/data_{degree}_{side}.pkl', 'rb') as data_input_file:
            K = pickle.load(data_input_file)
            R = pickle.load(data_input_file)
            t = pickle.load(data_input_file)

        render_depth = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)
        renders[image] = {'K': K, 'R': R, 't': t, 'render': render_depth}
    return renders


renders_matrices = load_render()


def silhouette_sampling(point):
    x, y, z = point
    for image in list(filter(lambda img: 'right' in img, images)):
        K = renders_matrices[image]['K']
        R = renders_matrices[image]['R']
        t = renders_matrices[image]['t']
        render_depth = renders_matrices[image]['render']

        p = project_point([x, y, z], R, t, K)
        # render_depth = renders_matrices[image]  # cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)
        # render = np.array(render_depth[:, :, 0:3])
        depth = render_depth[:, :, 3]
        '''
        if debug:
            cv.drawMarker(render, p, [0, 255, 0], cv.MARKER_CROSS, 2, 1)
            cv.imshow('foobar', render)
            cv.waitKey(1)
        '''

        is_outside = p[0] < 0 or p[0] >= 256 or p[1] < 0 or p[1] >= 256
        if is_outside or depth[p[1], p[0]] == 0:
            return 1
    return -1


def laser_ray_sampling(image, laser_points):
    points = []
    degree = int(image.split('_')[1])
    side = image.split('_')[2]
    with open(f'renders/data_{degree}_{side}.pkl',
              'rb') as data_input_file:
        K = pickle.load(data_input_file)
        R = pickle.load(data_input_file)
        t = pickle.load(data_input_file)
        laser_center = pickle.load(data_input_file)
        laser_norm = pickle.load(data_input_file)

    a, b, c = laser_norm
    d = -(a * laser_center[0] + b * laser_center[1] + c * laser_center[2])

    render = np.array(cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)[:, :, 0:3])
    red_channel = render[:, :, 2] * 255
    _, red_channel = cv.threshold(red_channel, 100, 255, cv.THRESH_BINARY)

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))
    point_cloud_e = [laser_center, camera_position]
    point_cloud_u = []  # [laser_center, camera_position]

    for u in range(red_channel.shape[0]):
        for v in range(red_channel.shape[1]):
            if not red_channel[u][v]:
                continue

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
            ), -1])
            break

    for _ in range(laser_points):  # here
        x, y, z = sample_point_from_plane([a, b, c, d], degree + (30 if side == 'right' else - 30), side)

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

        if debug:
            cv.drawMarker(render, p_far_point, [255, 255, 0], cv2.MARKER_DIAMOND, 2, 1)
            cv.drawMarker(render, p, [0, 0, 255], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)

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

    for point in [[random.uniform(-3, 3), random.uniform(-3, 0), random.uniform(-3, 3)] for _ in
                  range(silhouette_points)]:
        label = silhouette_sampling(point)
        if label == 1:
            external.append(point)
        else:
            unknown.append(point)

    sampling_list = images.copy()
    for _ in range(360 * 2):
        image = random.sample(sampling_list, 1)[0]
        for point, label in laser_ray_sampling(image, laser_points):
            if label == 1:
                external.append(point)
            elif label == 0:
                unknown.append(point)
            else:
                internal.append(point)

    if debug:
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in unknown])
        point_cloud.plot(eye_dome_lighting=True)
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in external])
        point_cloud.plot(eye_dome_lighting=True)

    print("Uniform raw dataset created - executing KNN")
    external, internal = knn_point_classification(external, internal, unknown, 5)
    print("Uniform dataset created")

    if debug:
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in internal])
        point_cloud.plot(eye_dome_lighting=True)
        point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in external])
        point_cloud.plot(eye_dome_lighting=True)

    inputs = np.concatenate((inputs, [[p_p * 10 for p_p in p] for p in external]), axis=0)
    inputs = np.concatenate((inputs, [[p_p * 10 for p_p in p] for p in internal]), axis=0)

    labels = torch.tensor([[1] for _ in external] + [[-1] for _ in internal], dtype=torch.float32,
                          requires_grad=True, device=device)

    dataset = [
        [torch.tensor(torch.from_numpy(inputs[i]), dtype=torch.float32, requires_grad=True, device=device), labels[i]]
        for i
        in range(len(inputs))]
    return dataset  # torch.tensor(dataset, device=device)


def create_gradient_base_dataset(gradient_image, silhouette_points=3000, laser_points=300):
    external = []
    internal = []
    unknown = []

    # FIXME

    # print(uniform_dataset.data[0])
    '''
    if uniform_dataset is not None:
        for point, label in uniform_dataset.data:
            if label == 1:
                external.append(point.detach().cpu().numpy())
            else:
                internal.append(point.detach().cpu().numpy())
    '''

    inputs = np.array([]).reshape(0, 3)

    for point in sample_from_gradient_image(gradient_image, silhouette_points):
        label = silhouette_sampling([p_p / 10 for p_p in point])
        if label == 1:
            external.append(point)
        else:
            unknown.append(point)
    '''
    '''

    sampling_list = images.copy()
    for _ in range(360 * 2):
        image = random.sample(sampling_list, 1)[0]
        # FIXME: image = random.sample(sampling_list, 1)[0]
        # image = sampling_list[91 + 30]  # 90
        # print(image)
        for point, label in laser_ray_gradient_sampling(image, gradient_image, laser_points):
            if label == 1:
                external.append(point)
            elif label == 0:
                unknown.append(point)
            else:
                internal.append(point)
        # print(f"{image} done")
    '''
    '''

    if debug:
        point_cloud = pv.PolyData(unknown)
        point_cloud.plot(eye_dome_lighting=True)
        point_cloud = pv.PolyData(external)
        point_cloud.plot(eye_dome_lighting=True)

    external, internal = knn_point_classification(external, internal, unknown, 5)
    # external, internal = pure_knn_point_classification(external, internal, unknown, 5)

    if debug:
        point_cloud = pv.PolyData(internal)
        point_cloud.plot(eye_dome_lighting=True)
        point_cloud = pv.PolyData(external)
        point_cloud.plot(eye_dome_lighting=True)

    inputs = np.concatenate((inputs, external), axis=0)
    inputs = np.concatenate((inputs, internal), axis=0)

    labels = torch.tensor([[1] for _ in external] + [[-1] for _ in internal], dtype=torch.float32,
                          requires_grad=True, device=device)

    dataset = [
        [torch.tensor(torch.from_numpy(inputs[i]), dtype=torch.float32, requires_grad=True, device=device), labels[i]]
        for i
        in range(len(inputs))]
    return dataset  # torch.tensor(dataset, device=device)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label


if UNIFORM_TRAINING_EPOCHS > 0:
    uniform_dataset = CustomDataset(create_uniform_dataset(40000, 3000))
else:
    uniform_dataset = None


def train_one_epoch_uniformly(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # uniform_dataset.data.to(device)
    training_loader = torch.utils.data.DataLoader(uniform_dataset, batch_size=256, shuffle=True,
                                                  generator=torch.Generator(device=device), num_workers=0)
    for batch_index, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TODO print(f"batch {batch_index} done, loss: {loss.item()}", end='\r')

        if batch_index == len(training_loader) - 1:
            last_loss = running_loss / len(training_loader)  # loss per batch
            print('  batch {} loss: {}'.format(batch_index + 1, last_loss))
            tb_x = epoch_index * 100 + batch_index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


for epoch in range(UNIFORM_TRAINING_EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch_uniformly(epoch_number, writer)

    running_vloss = 0.0
    '''
    model.eval()
    
    with torch.no_grad():
        val_x = np.random.uniform(-30., 30., 100)
        val_y = np.random.uniform(-30., 0., 100)
        val_z = np.random.uniform(-30., 30, 100)

        v_inputs = [[val_x[i], val_y[i]] for i in range(100)]
        v_labels = torch.tensor([[realistic_oracle(p)] for p in v_inputs], dtype=torch.float32, requires_grad=True,
                                device=device)

        v_outputs = model(torch.tensor(v_inputs, dtype=torch.float32, requires_grad=True, device=device))
        v_loss = loss_fn(v_outputs, v_labels)

    print('LOSS train {} valid {}'.format(avg_loss, v_loss))
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


def compute_gradient_image_from_model():
    # gradient_image = torch.empty(0, dtype=torch.float32,
    #                             device=device)  # torch.tensor([], dtype=torch.float32,
    #            device=device)  # torch.zeros([100, 50, 100], dtype=torch.float32, device=device)  # np.zeros([100, 50, 100])

    x = torch.linspace(-30, 30, 100, device=device)
    y = torch.linspace(-30, 0, 50, device=device)
    z = torch.linspace(-30, 30, 100, device=device)
    X, Y, Z = torch.meshgrid(x, y, z)

    '''
    cpu_model = INR3D('cpu')
    cpu_model.cpu()
    cpu_model.load_state_dict(torch.load('3d-model-stable', map_location='cpu'))
    
    points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)
    # return model(points).view(100, 50, 100)

    for [x, y, z] in points:
        input_tensor = torch.tensor([[x, y, z]], dtype=torch.float32, requires_grad=True, device=device)
        output = model(input_tensor)
        # grad_outputs = torch.ones_like(output[0])
        # return torch.autograd.grad(outputs=output, inputs=input_tensor, grad_outputs=grad_outputs,
        #                           is_grads_batched=True)
        output.backward()
        gradient_x, gradient_y, gradient_z = input_tensor.grad[0]

        gradient_image[
            closest(torch.linspace(-30, 30, 100), x),
            closest(torch.linspace(-30, 0, 50), y),
            closest(torch.linspace(-30, 30, 100), z)
        ] = math.sqrt((gradient_x ** 2) + (gradient_y ** 2) + (gradient_z ** 2))
    # print([x, y, z])
    '''

    points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)
    # return model(points).view(100, 50, 100)

    print("start")
    input_tensor = torch.tensor(points, dtype=torch.float32, requires_grad=True, device=device)
    output = model(input_tensor)
    grad_outputs = torch.autograd.grad(outputs=output, inputs=input_tensor, grad_outputs=torch.ones_like(output),
                                       is_grads_batched=False)[0]
    # output.backward()
    print("mid")

    def foo(elem):
        print("hej")
        print(elem)
        return elem

    # torch.vmap(foo)(grad_outputs)

    if grad_outputs.device != 'cpu':
        grad_outputs.detach().cpu()

    gradient_image = torch.tensor([math.sqrt((x ** 2) + (y ** 2) + (z ** 2)) for [x, y, z] in grad_outputs],
                                  dtype=torch.float32, device='cpu')

    gradient_image = gradient_image.to(device)
    print("end")

    return gradient_image  # torch.tensor(gradient_image, dtype=torch.float32, device=device)  # model(points).view(100, 50, 100)


def sample_from_gradient_image(gradient_image, k):
    points = []
    x_distribution, _ = margins(gradient_image[:, 25, :])

    for _ in range(k):
        x = x_distribution.flatten()
        x /= np.sum(x)
        cdf = np.cumsum(x)

        '''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Marginal X')

        ax1.step(torch.linspace(-30, 30, 100), x, where='mid', label='CDF')
        ax1.grid(True)
        ax2.step(torch.linspace(-30, 30, 100), cdf, where='mid', label='CDF')
        ax2.grid(True)
        plt.show()
        '''

        probabilities_to_invert = np.random.uniform(0, 1, 1)
        x = closest(torch.linspace(-30, 30, 100),
                    [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

        y_distribution, _ = margins(gradient_image[x, :, :])
        y = y_distribution.flatten()
        y /= np.sum(y)
        cdf = np.cumsum(y)

        '''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Marginal Y')

        ax1.step(torch.linspace(-30, 0, 50), y, where='mid', label='CDF')
        ax1.grid(True)
        ax2.step(torch.linspace(-30, 0, 50), cdf, where='mid', label='CDF')
        ax2.grid(True)
        plt.show()
        '''

        probabilities_to_invert = np.random.uniform(0, 1, 1)
        y = closest(torch.linspace(-30, 0, 50),
                    [inverse_cdf(p, torch.linspace(-30, 0, 50), cdf) for p in probabilities_to_invert][0])

        _, z_distribution = margins(gradient_image[:, y, :])

        z = z_distribution.flatten()
        z /= np.sum(z)
        cdf = np.cumsum(z)

        '''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Marginal Z')

        ax1.step(torch.linspace(-30, 30, 100), z, where='mid', label='CDF')
        ax1.grid(True)
        ax2.step(torch.linspace(-30, 30, 100), cdf, where='mid', label='CDF')
        ax2.grid(True)
        plt.show()
        '''

        probabilities_to_invert = np.random.uniform(0, 1, 1)
        z = closest(torch.linspace(-30, 30, 100),
                    [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

        # print([torch.linspace(-30, 30, 100)[x], torch.linspace(-30, 0, 50)[y], torch.linspace(-30, 30, 100)[z]])
        points.append([torch.linspace(-30, 30, 100)[x].item(), torch.linspace(-30, 0, 50)[y].item(),
                       torch.linspace(-30, 30, 100)[z].item()])

        x_distribution, _ = margins(gradient_image[:, :, z])
        # x_distribution, _ = margins(distribution)[:, :, z]

    # point_cloud = pv.PolyData(points)
    # point_cloud.plot(eye_dome_lighting=True)
    '''

    cdf = np.cumsum(x)

    # Plot the CDF
    plt.step(torch.linspace(-30, 30, 100), cdf, where='mid', label='CDF')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    return points


def laser_ray_gradient_sampling(image, gradient_image, laser_points=300):
    points = []
    degree = int(image.split('_')[1])
    side = image.split('_')[2]
    with open(f'renders/data_{degree}_{side}.pkl',
              'rb') as data_input_file:
        K = pickle.load(data_input_file)
        R = pickle.load(data_input_file)
        t = pickle.load(data_input_file)
        laser_center = pickle.load(data_input_file)
        laser_norm = pickle.load(data_input_file)

    a, b, c = laser_norm
    d = -(a * laser_center[0] + b * laser_center[1] + c * laser_center[2])

    render = np.array(cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)[:, :, 0:3])
    red_channel = render[:, :, 2] * 255
    _, red_channel = cv.threshold(red_channel, 100, 255, cv.THRESH_BINARY)

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))
    point_cloud_e = [laser_center, camera_position]
    point_cloud_u = []  # [laser_center, camera_position]

    for u in range(red_channel.shape[0]):
        for v in range(red_channel.shape[1]):
            if not red_channel[u][v]:
                continue

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
            ), -1])
            break

    # print(degree)
    sampled_points = sample_point_from_plane_gradient([a, b, c, d],
                                                      degree + (30 if side == 'right' else - 30),
                                                      model, laser_points)

    '''
    plt.figure()
    plt.imshow(gradient_image[:, :, 50])
    plt.show(block=False)

    for point in sampled_points:  # .flatten(start_dim=0, end_dim=1):
        p = project_point(point.numpy().tolist(), R, t, K)
        cv.drawMarker(render, p, [255, 255, 0], cv.MARKER_DIAMOND, 2, 1)
        cv.imshow("test", render)
        cv.waitKey(1)

    cv.waitKey(0)
    '''

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

        if debug:
            cv.drawMarker(render, p_far_point, [255, 255, 0], cv2.MARKER_DIAMOND, 2, 1)
            cv.drawMarker(render, p, [0, 0, 255], cv.MARKER_CROSS, 2, 2)
            cv.imshow('red', red_channel)
            cv.imshow('foobar', render)
            cv.waitKey(1)

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


def train_one_epoch_gradient(epoch_index, tb_writer):
    print("gradient image done")
    running_loss = 0.
    last_loss = 0.

    training_loader = torch.utils.data.DataLoader(gradient_based_dataset, batch_size=128, shuffle=True,
                                                  generator=torch.Generator(device=device), num_workers=0)
    for batch_index, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
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
torch.save(model.state_dict(), '3d-model')
print("Uniform training completed, intermediate model saved")
model.train(True)

if GRADIENT_BASED_TRAINING_EPOCHS > 0:
    test = compute_gradient_image_from_model().view(100, 50, 100)
    if device.type != 'cpu':
        test = test.cpu().detach().numpy()

    gradient_based_dataset = CustomDataset(create_gradient_base_dataset(test, 10000, 1000))

else:
    test = None

for epoch in range(GRADIENT_BASED_TRAINING_EPOCHS):
    print('EPOCH - gradient based - {}:'.format(epoch_number + 1))
    # if (epoch - UNIFORM_TRAINING_EPOCHS) == GRADIENT_BASED_TRAINING_EPOCHS / 2:
    #    gradient_based_dataset = CustomDataset(create_gradient_base_dataset(test, 3000, 300))

    if debug:
        plane = test[:, 25, :]
        fig = plt.figure()
        plt.imshow(plane)
        plt.show(block=True)

        plane = test[50, :, :]
        fig = plt.figure()
        plt.imshow(plane)
        plt.show(block=True)

        plane = test[:, :, 50]
        fig = plt.figure()
        plt.imshow(plane)
        plt.show(block=True)

    # print(test)
    avg_loss = train_one_epoch_gradient(epoch_number, writer)

    running_vloss = 0.0

    '''
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
