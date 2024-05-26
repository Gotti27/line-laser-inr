import argparse
import os
import pickle
from datetime import datetime

import cv2
import pyvista as pv
import torch
from torch.utils.tensorboard import SummaryWriter

from inr_model import INR3D
from utils import *

UNIFORM_TRAINING_EPOCHS = 7
GRADIENT_BASED_TRAINING_EPOCHS = 5
# INTRA_RAY_DEGREES = 1
UNIFORM_BATCH_NUMBER = 50
GRADIENT_BASED_BATCH_NUMBER = 10

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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


def sample_from_image(image):
    internal = []
    external = []
    unknown = []
    degree = int(image.split('_')[1])
    render = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)
    red_render = render[:, :, 2] * 255
    _, red_render = cv.threshold(red_render, 100, 255, cv.THRESH_BINARY)

    with open(f'renders/data_{degree}.pkl', 'rb') as data_input_file:
        K = pickle.load(data_input_file)
        R = pickle.load(data_input_file)
        t = pickle.load(data_input_file)
        laser_center = pickle.load(data_input_file)
        laser_norm = pickle.load(data_input_file)

    a, b, c = laser_norm
    d = -(a * laser_center[0] + b * laser_center[1] + c * laser_center[2])

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))

    laser_points = []
    for u in range(red_render.shape[0]):
        for v in range(red_render.shape[1]):
            if not red_render[u][v]:
                continue
            laser_point_camera = np.array(
                [v - (red_render.shape[1] / 2), u - (red_render.shape[0] / 2), K[0][0], 1])
            laser_point_world = np.concatenate([
                np.concatenate([R.T, np.array(- R.T @ t).reshape(3, 1)], axis=1),
                np.array([[0, 0, 0, 1]])
            ], axis=0) @ laser_point_camera

            laser_point_world = [laser_point_world[0] / laser_point_world[3],
                                 laser_point_world[1] / laser_point_world[3],
                                 laser_point_world[2] / laser_point_world[3]]

            laser_points.append(np.squeeze(
                np.asarray(find_plane_line_intersection([a, b, c, d], camera_position, np.array(laser_point_world)))
            ))
            break

    for point in laser_points:
        internal.append(point)
        if debug:
            point = np.append(point, [1])  # = np.hstack([point, [[1.]]])
            p = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ point
            p = [int(round(p[0, 0] / p[0, 2])), int(round(p[0, 1] / p[0, 2]))]
            cv.drawMarker(render, p, [0, 255, 0], cv.MARKER_TILTED_CROSS, 1, 1)

            p_laser_center = laser_center @ rotate_y(0)
            p_laser_center = project_point(p_laser_center, R, t, K)

            cv.line(render, p_laser_center, p, [0, 255, 0], 2)

    for laser_point in laser_points:
        for _ in range(5):
            if 45 <= degree + 30 < 135:
                x = random.uniform(laser_point[0], 3.5)
                y = laser_point[1]
                z = -(a * x + b * y + d) / c
            elif 135 <= degree + 30 < 225:
                z = random.uniform(laser_point[2], 3.5)
                y = laser_point[1]
                x = -(c * z + b * y + d) / a
            elif 225 <= degree + 30 < 315:
                x = random.uniform(-3.5, laser_point[0])
                y = laser_point[1]
                z = -(a * x + b * y + d) / c
            else:
                z = random.uniform(-3.5, laser_point[2])
                y = laser_point[1]
                x = -(c * z + b * y + d) / a

            external.append([x, y, z])
            if debug:
                p = np.array([x, y, z, 1.])
                p = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p
                p = [int(round(p[0, 0] / p[0, 2])), int(round(p[0, 1] / p[0, 2]))]
                cv.drawMarker(render, p, [255, 255, 0], cv.MARKER_TILTED_CROSS, 2, 1)

        for _ in range(5):
            if 45 <= degree + 30 < 135:
                x = random.uniform(-3.5, laser_point[0])
                y = laser_point[1]
                z = -(a * x + b * y + d) / c
            elif 135 <= degree + 30 < 225:
                z = random.uniform(-3.5, laser_point[2])
                y = laser_point[1]
                x = -(c * z + b * y + d) / a
            elif 225 <= degree + 30 < 315:
                x = random.uniform(laser_point[0], 3.5)
                y = laser_point[1]
                z = -(a * x + b * y + d) / c
            else:
                z = random.uniform(laser_point[2], 3.5)
                y = laser_point[1]
                x = -(c * z + b * y + d) / a

            unknown.append([x, y, z])
            if debug:
                p = np.array([x, y, z, 1.])
                p = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p
                p = [int(round(p[0, 0] / p[0, 2])), int(round(p[0, 1] / p[0, 2]))]
                cv.drawMarker(render, p, [255, 0, 255], cv.MARKER_TILTED_CROSS, 2, 1)

    if debug:
        laser_points = []
        for p in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], (laser_norm + np.array([0, 0, 0.])).tolist()]:
            p.append(1)
            camera_p = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p

            laser_points.append(
                [camera_p[0, 0] / camera_p[0, 2], camera_p[0, 1] / camera_p[0, 2]]
            )

        origin = laser_points[0]
        top_x = laser_points[1]
        top_y = laser_points[2]
        top_z = laser_points[3]
        norm_test = laser_points[5]

        cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_x[0])), int(round(top_x[1]))],
                [0, 0, 255], 1)
        cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_y[0])), int(round(top_y[1]))],
                [0, 255, 0], 1)
        cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_z[0])), int(round(top_z[1]))],
                [255, 0, 0], 1)
        cv.line(render, [int(round(origin[0])), int(round(origin[1]))],
                [int(round(norm_test[0])), int(round(norm_test[1]))],
                [0, 255, 255], 1)

        cv.imshow('foo', render)
        cv.waitKey(1)

    return external, internal, unknown


def silhouette_sampling(point):
    x, y, z = point
    for image in list(filter(lambda img: 'right' in img, images)):
        degree = image.split('_')[1]
        side = image.split('_')[2]
        with open(f'renders/data_{degree}_{side}.pkl', 'rb') as data_input_file:
            K = pickle.load(data_input_file)
            R = pickle.load(data_input_file)
            t = pickle.load(data_input_file)

        p = np.array([x, y, z, 1.])
        p = K @ np.concatenate([R, np.matrix(t).T], axis=1) @ p
        p = [int(round(p[0, 0] / p[0, 2])), int(round(p[0, 1] / p[0, 2]))]
        render_depth = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)
        render = np.array(render_depth[:, :, 0:3])
        depth = render_depth[:, :, 3]
        if debug:
            cv.drawMarker(render, p, [0, 255, 0], cv.MARKER_CROSS, 2, 1)
            cv.imshow('foobar', render)
            cv.waitKey(1)

        is_outside = p[0] < 0 or p[0] >= 256 or p[1] < 0 or p[1] >= 256

        if is_outside or depth[p[1], p[0]] == 0:
            return 1
    return -1


def laser_ray_sampling(image):
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

    for _ in range(100):
        if side == 'right':
            x, y, z = sample_point_from_plane([a, b, c, d], degree + 30)
        else:
            x, y, z = sample_point_from_plane([a, b, c, d], degree - 30)

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

    return points


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for batch_index in range(UNIFORM_BATCH_NUMBER):
        external = []
        internal = []
        unknown = []

        inputs = np.array([]).reshape(0, 3)

        for point in [[random.uniform(-3, 3), random.uniform(-3, 0), random.uniform(-3, 3)] for _ in
                      range(100)]:
            label = silhouette_sampling(point)
            if label == 1:
                external.append(point)
            else:
                unknown.append(point)

        sampling_list = images.copy()
        for _ in range(360 * 2):
            image = random.sample(sampling_list, 1)[0]
            for point, label in laser_ray_sampling(image):
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

        external, internal = knn_point_classification(external, internal, unknown, 5)

        if debug:
            point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in internal])
            point_cloud.plot(eye_dome_lighting=True)
            point_cloud = pv.PolyData([[p_p * 10 for p_p in p] for p in external])
            point_cloud.plot(eye_dome_lighting=True)

        inputs = np.concatenate((inputs, [[p_p * 10 for p_p in p] for p in external]), axis=0)
        inputs = np.concatenate((inputs, [[p_p * 10 for p_p in p] for p in internal]), axis=0)

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
