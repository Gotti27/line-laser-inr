import argparse
import math
import os
import random
from datetime import datetime

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv

from src.dataset import load_renders
from src.utils import project_point, find_plane_line_intersection_2, rotate_z

target = 'Dragon-small'
NUMBER_IMAGES = 1000
# [20, 50, 100, 150, 200, 300, 400, 500, 600, 1000]

np.bool = np.bool_
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", dest="debug", help="Enable debug mode", action="store_true", default=False)
args = parser.parse_args()

debug = args.debug
if debug:
    print("---DEBUG MODE ACTIVATED---")

print(f"Started {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")

image_folder = f'renders/{target}'
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images = random.sample(images, NUMBER_IMAGES if 0 < NUMBER_IMAGES < len(images) else len(images))
images.sort(key=lambda name: int(name.split('_')[1]))

point_cloud_file = open(f"pointcloud-{target}.xyz", "w")  # a+

renders_matrices = load_renders(images, target)

for image in images:
    degree = int(image.split('_')[1])
    side = image.split('_')[2]

    K = renders_matrices[image]['K']
    R = renders_matrices[image]['R']
    t = renders_matrices[image]['t']
    laser_center = renders_matrices[image]['laser_center']
    laser_norm = renders_matrices[image]['laser_norm']

    # laser_center = [0, 0, 0]
    a, b, c = laser_norm
    d = -(a * laser_center[0] + b * laser_center[1] + c * laser_center[2])

    render = renders_matrices[image]['render'][:, :, 0:3]
    red_channel = render[:, :, 2] * 255
    _, red_channel = cv.threshold(red_channel, 100, 255, cv.THRESH_BINARY)

    camera_position = np.squeeze(np.asarray(- np.matrix(R).T @ t))
    render = np.array(render)

    for u in range(render.shape[0]):
        last = None
        for v in range(render.shape[1]):
            if not red_channel[u, v]:
                continue
            last = [u, v]

        if last is None:
            continue
        u, v = last
        # cv.drawMarker(render, [v, u], [255, 255, 0], cv.MARKER_TILTED_CROSS, 1, 1)

        laser_point_camera = np.array(
            [v - (red_channel.shape[1] / 2), u - (red_channel.shape[0] / 2), K[0][0],
             1])

        laser_point_world = np.concatenate([
            np.concatenate([R.T, np.array(- R.T @ t).reshape(3, 1)], axis=1),
            np.array([[0, 0, 0, 1]])
        ], axis=0) @ laser_point_camera

        laser_point_world = [laser_point_world[0] / laser_point_world[3],
                             laser_point_world[1] / laser_point_world[3],
                             laser_point_world[2] / laser_point_world[3]]

        world_point = np.squeeze(
            np.asarray(
                find_plane_line_intersection_2(laser_norm, laser_center, camera_position, np.array(laser_point_world)))
        )

        cv.drawMarker(render, project_point(world_point.tolist(), R, t, K), [0, 255, 0], cv.MARKER_TILTED_CROSS, 1,
                      1)
        # world_point *= 10

        world_point @= rotate_z(180)
        '''
        world_point @= rotate_y(180)
        world_point @= rotate_x(90)
        '''

        point_cloud_file.write(f"{world_point[0]} {world_point[1]} {world_point[2]}\n")

    if debug:
        cv.imshow('foobar', render)
        cv.waitKey(1)

point_cloud_file.close()
point_cloud = o3d.io.read_point_cloud(f"pointcloud-{target}.xyz")
ground_truth = o3d.io.read_point_cloud(f"scenes/meshes/{target}.ply")

if debug:
    o3d.visualization.draw_geometries([ground_truth, point_cloud])

o3d.geometry.PointCloud.estimate_normals(point_cloud)
point_cloud.orient_normals_consistent_tangent_plane(20)
o3d.io.write_point_cloud(f'models/{target}-pointcloud.ply', point_cloud)
# point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=100))
if debug:
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=point_cloud, depth=6)


def mesh_to_cloud_signed_distances(o3d_mesh: o3d.t.geometry.TriangleMesh,
                                   cloud: o3d.t.geometry.PointCloud) -> np.ndarray:
    # cloud.estimate_normals()
    # cloud.orient_normals_consistent_tangent_plane(50)
    mesh_raycast = o3d.t.geometry.RaycastingScene()
    mesh_id = mesh_raycast.add_triangles(o3d_mesh)

    points = np.asarray(cloud.point.positions[mesh_points_indexes])
    normals = np.asarray(cloud.point.normals[mesh_points_indexes])

    distances = []

    for i in range(len(points)):
        ray_origin = points[i]
        ray_direction = normals[i]

        ray_positive = mesh_raycast.cast_rays(
            o3d.core.Tensor([i.numpy().tolist() for i in ray_origin] + [i.numpy().tolist() for i in ray_direction],
                            dtype=o3d.core.Dtype.Float32).reshape((1, 6)))
        ray_negative = mesh_raycast.cast_rays(
            o3d.core.Tensor([i.numpy().tolist() for i in ray_origin] + [- i.numpy().tolist() for i in ray_direction],
                            dtype=o3d.core.Dtype.Float32).reshape((1, 6)))

        distance_positive = ray_positive['t_hit'].numpy()
        distance_negative = ray_negative['t_hit'].numpy()

        distance = min(abs(distance_positive), abs(distance_negative))

        if np.isinf(distance):
            distance = np.nan

        '''
        if len(distance_positive) > 0 and distance_positive[0] >= 0 and np.isfinite(distance_positive[0]):
            distance = distance_positive[0]
        elif len(distance_negative) > 0 and distance_negative[0] >= 0:
            distance = -distance_negative[0]
        else:
            distance = np.nan
        '''

        if isinstance(distance, np.ndarray):
            distance = distance.tolist()[0]

        distances.append(distance)

    return np.array(distances)


mesh.compute_vertex_normals()
mesh.vertex_colors = o3d.utility.Vector3dVector(
    np.array([[200., 200., 0.]]))

if debug:
    o3d.visualization.draw_geometries([mesh])

mesh_points_indexes = [i for i in range(len(ground_truth.points)) if True]  # ground_truth.points[i][1] > 0.1]

###
distances_np = mesh_to_cloud_signed_distances(
    o3d.t.geometry.TriangleMesh.from_legacy(mesh),
    o3d.t.geometry.PointCloud.from_legacy(ground_truth))

distances_np *= 10

print(math.sqrt(np.sum(np.square(distances_np[~np.isnan(distances_np)]) / len(distances_np[~np.isnan(distances_np)]))))

# distances_normalized = (distances_np - distances_np.min()) / (distances_np.max() - distances_np.min())
'''
colormap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=0, vmax=2, clip=True)
distances_np = norm(distances_np)
colors = colormap(distances_np)[:, :3]

# ground_truth.colors = o3d.utility.Vector3dVector(colors)
ground_truth.colors = o3d.utility.Vector3dVector(
    np.array(colors)
)  # o3d.utility.Vector3dVector(np.random.rand(len(ground_truth.points), 3))
'''

if debug:
    o3d.visualization.draw_geometries([ground_truth])

vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)

faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

mesh_pv = pv.PolyData(vertices, faces_pv)

camera_position = [(100., 100., 55),
                   (-0.05642535239457658, -0.3681839525699573, 20.08591179996729),
                   (0.0, 0.0, 1.0)]

mesh_pv = mesh_pv.scale(10)
mesh_pv = mesh_pv.rotate_z(0)
mesh_pv.save(f'models/output-meshes/{target}/poisson.ply')

plotter = pv.Plotter()
plotter.add_mesh(mesh_pv, show_edges=False, color='lightblue')
plotter.camera_position = camera_position
# plotter.zoom_camera(1.6)
plotter.show_axes()
# plotter.save_graphic('test2.svg', title='PyVista Export', raster=True, painter=True)
plotter.enable_eye_dome_lighting()
plotter.show()
plotter.screenshot('point-based.png', False)

#####

points = np.asarray(ground_truth.points)
point_cloud_pv = pv.PolyData(points)

point_cloud_pv['errors'] = [min(e, 2.) for e in distances_np]
cmap = plt.cm.plasma
cmap.set_bad(color='green')

'''
if ground_truth.has_colors():
    colors = np.asarray(ground_truth.colors)
else:
    colors = None


if colors is not None:
    point_cloud_pv["colors"] = (colors * 255).astype(np.uint8)
'''

plotter = pv.Plotter()
point_cloud_pv = point_cloud_pv.scale(10)
point_cloud_pv = point_cloud_pv.rotate_z(140)
plotter.add_mesh(point_cloud_pv, scalars='errors', cmap=cmap, nan_color='green', point_size=10)
plotter.camera_position = camera_position
plotter.show_axes()
# plotter.save_graphic('test2.svg', title='PyVista Export', raster=True, painter=True)
plotter.show()
plotter.screenshot('point-based-error.png', False)
