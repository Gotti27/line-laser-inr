import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import open3d as o3d
import torch

from src.dataset import load_renders
from src.utils import project_point, find_plane_line_intersection, sample_point_from_plane, bresenham, \
    pure_knn_point_classification_eval, rotate_z, rotate_x

np.bool = np.bool_
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

target = 'igea'
device = 'cpu'
k_values = [3, 5, 7, 15, 23, 35, 53, 75, 93, 127]
iteration = 5
image_folder = f'renders/{target}'
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images.sort(key=lambda name: int(name.split('_')[1]))

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
    unknown = []

    pool = multiprocess.Pool()
    labels = pool.map(lambda p: [p, silhouette_sampling(p)],
                      [[random.uniform(-4, 4), random.uniform(-4, 0), random.uniform(-4, 4)] for _ in
                       range(silhouette_points)])

    for point, label in labels:
        if label == 1:
            external.append(point)
        else:
            unknown.append(point)

    sampling_list = images.copy()
    for _ in range(len(sampling_list)):
        image = random.sample(sampling_list, 1)[0]
        sampling_list.remove(image)
        to_check = []
        for point, label in laser_ray_sampling(image, laser_points):
            if label == 1:
                external.append(point)
            elif label == 0:
                to_check.append(point)
            else:
                pass
                # internal.append(point)

        labels = pool.map(lambda p: [p, silhouette_sampling(p)], to_check)
        for point, label in labels:
            if label == 1:
                external.append(point)
            else:
                unknown.append(point)

    return external, unknown


mesh = o3d.io.read_triangle_mesh(f"scenes/meshes/{target}.ply")
mesh.rotate(rotate_z(180), center=(0, 0, 0))
mesh.rotate(rotate_x(90), center=(0, 0, 0))

vertices = o3d.cpu.pybind.core.Tensor(np.asarray(mesh.vertices))
triangles = o3d.cpu.pybind.core.Tensor(np.asarray(mesh.triangles))
'''
#####
pointcloud = o3d.cpu.pybind.geometry.PointCloud(
    o3d.cpu.pybind.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float32)))
pointcloud.estimate_normals()

new_mesh = o3d.cpu.pybind.t.geometry.TriangleMesh(np.asarray(mesh.vertices, dtype=np.float32),
                                                  np.asarray(mesh.triangles)
                                                  )
new_mesh.compute_vertex_normals()
watertight_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud
                                                                                       , depth=8)
# bbox = watertight_mesh.get_axis_aligned_bounding_box()
# watertight_mesh = watertight_mesh.crop(bbox)
#####

# mesh = mesh.scale(10)
# mesh.compute_vertex_normals()
watertight_mesh = o3d.cpu.pybind.t.geometry.TriangleMesh(
    np.asarray(watertight_mesh.vertices, dtype=np.float32),
    np.asarray(watertight_mesh.triangles)
)
'''

scene = o3d.cpu.pybind.t.geometry.RaycastingScene()
scene.add_triangles(
    o3d.cpu.pybind.t.geometry.TriangleMesh(np.asarray(mesh.vertices, dtype=np.float32),
                                           np.asarray(mesh.triangles)
                                           )
)


# points = create_uniform_dataset(100000, 100)


def evaluate_k(k):
    external, internal = pure_knn_point_classification_eval(external_glob, [], unknown_glob, k)

    inputs = np.array([]).reshape(0, 3)
    inputs = np.concatenate((inputs, external), axis=0, dtype=np.float32)
    inputs = np.concatenate((inputs, internal), axis=0, dtype=np.float32)

    labels = torch.tensor([[1] for _ in external] + [[-1] for _ in internal], dtype=torch.float32,
                          requires_grad=True, device=device)

    signed_distance = scene.compute_signed_distance(o3d.cpu.pybind.core.Tensor(inputs))

    same = 0
    diff = 0

    true_positive = 0
    false_positive = 0
    false_negatives = 0

    val_internal = []
    val_external = []

    for i, dist in enumerate(signed_distance):
        if dist <= 0:
            val_internal.append(inputs[i])
        else:
            val_external.append(inputs[i])

        if (dist * labels[i]) > 0 or (dist == 0 and labels[i] == 0):
            same += 1
            if dist <= 0:
                true_positive += 1
        else:
            diff += 1
            if dist < 0:
                false_negatives += 1
            elif dist > 0:
                false_positive += 1

    '''
    plotter = pv.Plotter()
    plotter.add_points(pv.PolyData(external), color='red')
    plotter.add_points(pv.PolyData(internal), color='blue')
    # faces = np.asarray(mesh.triangles)
    # faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    # pv_mesh = pv.PolyData(np.asarray(mesh.vertices), faces_pv)
    # plotter.add_mesh(pv_mesh)
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_points(pv.PolyData(val_external), color='red')
    plotter.add_points(pv.PolyData(val_internal), color='blue')
    plotter.show()
    '''

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negatives)

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"same: {same / len(signed_distance)}")
    print(f"diff: {diff / len(signed_distance)}")
    return precision, recall


prec = []
rec = []
for _ in range(iteration):
    external_glob, unknown_glob = create_uniform_dataset(100000, 100)
    external_glob = [[p_p * 1 for p_p in p] for p in external_glob]
    unknown_glob = [[p_p * 1 for p_p in p] for p in unknown_glob]

    points = np.array([]).reshape(0, 3)
    points = np.concatenate((points, external_glob), axis=0, dtype=np.float32)
    points = np.concatenate((points, unknown_glob), axis=0, dtype=np.float32)

    print(f"unknown points: {len(unknown_glob)}")

    res_prec = []
    res_rec = []
    for k_val in k_values:
        p, r = evaluate_k(k_val)
        res_prec.append(p)
        res_rec.append(r)
    prec.append(res_prec)
    rec.append(res_rec)

np.savetxt(f"prec-multi-{target}.csv", prec, delimiter=",")
np.savetxt(f"rec-multi-{target}.csv", rec, delimiter=",")

plt.figure()

plt.errorbar(k_values,
             np.mean([p for p in prec], axis=0),
             np.std([p for p in prec], axis=0), fmt='o-b', capsize=4,
             label='Uniform sampling')

# plt.plot(k_values, [p[0] for p in prec_rec], marker='o', linestyle='', color='b')

'''
for i, label in enumerate(k_values):
    plt.text([p[1] for p in prec_rec][i], [p[0] for p in prec_rec][i], label, fontsize=9, ha='right')
'''

# plt.title("Precision-Recall for Different Configurations")
plt.xlabel("K")
plt.ylabel("Precision")
plt.grid(True)

plt.show()

plt.figure()

plt.errorbar(k_values,
             np.mean([p for p in rec], axis=0),
             np.std([p for p in rec], axis=0), fmt='o-b', capsize=4,
             label='Uniform sampling')

# plt.plot(k_values, [p[1] for p in prec_rec], marker='o', linestyle='', color='b')
'''
for i, label in enumerate(k_values):
    plt.text([p[1] for p in prec_rec][i], [p[0] for p in prec_rec][i], label, fontsize=9, ha='right')
'''

# plt.title("Precision-Recall for Different Configurations")
plt.xlabel("K")
plt.ylabel("Recall")
plt.grid(True)

plt.show()
