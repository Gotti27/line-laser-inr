import json
import os

import numpy as np
import pyvista as pv
import trimesh
import vortex as vx

from src.dataset import load_renders
from src.evaluation import metrics

gt_mesh = pv.read('scenes/meshes/bunny-small.ply')

dragon = pv.read('/Users/mario/Desktop/bunny.ply')
dragon = dragon.rotate_x(90)
gt_mesh = gt_mesh.rotate_x(90)
gt_mesh = gt_mesh.rotate_y(-90)
# gt_mesh = gt_mesh.scale(2)

pl = pv.Plotter()

target = 'Dragon-mid-no-laser'
image_folder = f'renders/{target}'
counter = 0
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images.sort(key=lambda name: int(name.split('_')[1]))

with open('data/transforms.json') as colmap_file:
    colmap = json.load(colmap_file)

poses = {}

for frame in colmap['frames']:
    rt = np.array(frame['transform_matrix'])
    r = rt[:3, :3]
    t = rt[:3, 3]
    camera_position = rt @ np.array([0, 0, 0, 1])
    camera_position = camera_position[:3]

    poses[frame['file_path'].split('/')[2].split('.')[0]] = {'colmap': camera_position}

    # camera_position = np.squeeze(np.asarray(-np.matrix(r).T @ t))

    pl.add_points(np.array([camera_position]), color='orange')

renders = load_renders(images, target, True)

for image in renders:
    print(image)
    roto_translation = np.concatenate(
        [np.concatenate([renders[image]['R'], np.matrix(renders[image]['t']).T], axis=1), np.matrix([0, 0, 0, 1])],
        axis=0)

    camera_position = np.squeeze(np.asarray(-np.matrix(renders[image]['R']).T @ renders[image]['t']))

    if image.split('.')[0] in poses:
        poses[image.split('.')[0]]['mitsuba'] = camera_position

    pl.add_points(np.array([camera_position]))
    # pl.add_points(np.array([camera_position]))
    # np.matrix(renders[image]['t'])

'''
pl.add_lines(np.array([[0, 0, 2.2], [0, 0, -2.2]]))
pl.add_lines(np.array([[2.2, 0, 0], [-2.2, 0, 0]]))
pl.add_lines(np.array([[0, 0, 0], [0, -2.2, 0]]))
pl.add_lines(np.array([np.array([2.2, 0, 0]) @ rotate_z(30), np.array([-2.2, 0, 0]) @ rotate_z(-30)]))
pl.add_lines(np.array([np.array([0, 0, 2.2]) @ rotate_x(-30), np.array([0, 0, -2.2]) @ rotate_x(30)]))
'''
pl.add_axes_at_origin()
pl.enable_eye_dome_lighting()
pl.add_mesh(dragon)
pl.add_mesh(gt_mesh)
pl.show()

rt, err = vx.geometry.absor(np.array([p['colmap'] for p in poses.values()]),
                            np.array([p['mitsuba'] for p in poses.values()]), True, True)
print(rt.to_4x4())

pl = pv.Plotter()
pl.add_axes_at_origin()
pl.show_grid()
pl.enable_eye_dome_lighting()
pl.add_points(rt.apply(np.array([p['colmap'] for p in poses.values()])), color='orange')
pl.add_points(np.array([p['mitsuba'] for p in poses.values()]))

dragon.points = rt.apply(dragon.points)

dragon = dragon.scale(0.5)  # fixme
pl.add_mesh(dragon)
pl.add_mesh(gt_mesh)

pl.show()

######################

vertices = dragon.points
faces = dragon.faces.reshape((-1, 4))[:, 1:]

num_points = 100000
points, _ = trimesh.sample.sample_surface(trimesh.Trimesh(vertices=vertices, faces=faces), num_points)

pl = pv.Plotter()
pl.add_axes_at_origin()
pl.show_grid()
pl.enable_eye_dome_lighting()
pl.add_points(points, color='orange')
pl.show()

dragon.points = rt.apply(dragon.points)

# dragon = dragon.scale(0.5)
pl.add_mesh(dragon)
pl.add_mesh(gt_mesh)

gt_vertices = gt_mesh.points
gt_faces = gt_mesh.faces.reshape((-1, 4))[:, 1:]

gt_points, _ = trimesh.sample.sample_surface(trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces), num_points)

print(metrics.chamfer_distance(points, gt_points))
print(metrics.hausdorff_distance(points, gt_points))
