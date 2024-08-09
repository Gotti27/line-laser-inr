import argparse
import math
import os
import random
from datetime import datetime

import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.dataset import load_renders
from src.utils import find_plane_line_intersection, project_point, rotate_y, rotate_x

target = 'Armadillo'
NUMBER_IMAGES = 0

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
images = random.sample(images, NUMBER_IMAGES if NUMBER_IMAGES > 0 else len(images))
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
            np.asarray(find_plane_line_intersection([a, b, c, d], camera_position, np.array(laser_point_world)))
        )

        cv.drawMarker(render, project_point(world_point.tolist(), R, t, K), [0, 255, 0], cv.MARKER_TILTED_CROSS, 1,
                      1)
        # world_point *= 10

        world_point @= rotate_y(180)
        world_point @= rotate_x(90)

        point_cloud_file.write(f"{world_point[0]} {world_point[1]} {world_point[2]}\n")

    cv.imshow('foobar', render)
    cv.waitKey(1)

point_cloud_file.close()
point_cloud = o3d.io.read_point_cloud(f"pointcloud-{target}.xyz")
ground_truth = o3d.io.read_point_cloud(f"scenes/meshes/{target}.ply")

o3d.visualization.draw_geometries([ground_truth, point_cloud])

o3d.geometry.PointCloud.estimate_normals(point_cloud)
point_cloud.orient_normals_consistent_tangent_plane(20)
# point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=100))
o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=point_cloud, depth=6)


def mesh_to_cloud_signed_distances(o3d_mesh: o3d.t.geometry.TriangleMesh,
                                   cloud: o3d.t.geometry.PointCloud) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)
    sdf = scene.compute_signed_distance(cloud.point.positions)
    return abs(sdf.numpy())


mesh.compute_vertex_normals()
mesh.vertex_colors = o3d.utility.Vector3dVector(
    np.array([[200., 200., 0.]]))

o3d.visualization.draw_geometries([mesh])

###
distances_np = np.asarray(mesh_to_cloud_signed_distances(
    o3d.t.geometry.TriangleMesh.from_legacy(mesh),
    o3d.t.geometry.PointCloud.from_legacy(ground_truth))
)

distances_np *= 10

print(math.sqrt(np.sum(np.square(distances_np) / len(distances_np))))

distances_normalized = (distances_np - distances_np.min()) / (distances_np.max() - distances_np.min())

colormap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
distances_np = norm(distances_np)
colors = colormap(distances_np)[:, :3]

# ground_truth.colors = o3d.utility.Vector3dVector(colors)
ground_truth.colors = o3d.utility.Vector3dVector(
    np.array(colors)
)  # o3d.utility.Vector3dVector(np.random.rand(len(ground_truth.points), 3))

o3d.visualization.draw_geometries([ground_truth])
