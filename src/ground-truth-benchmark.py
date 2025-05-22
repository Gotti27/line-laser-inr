import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import trimesh
from skimage import measure

from inr_model import INR3D
from src import utils

np.bool = np.bool_

mesh = pv.read(f'scenes/meshes/Dragon-small.ply')  # pv.examples.download_dragon()
# mesh.compute_normals(inplace=True)
mesh = mesh.translate([-dim for dim in mesh.center])
x_length = mesh.bounds[1] - mesh.bounds[0]
y_length = mesh.bounds[3] - mesh.bounds[2]
z_length = mesh.bounds[5] - mesh.bounds[4]

max_length = max(x_length, y_length, z_length)
scaling_factor = 1 / max_length

mesh = mesh.scale([scaling_factor, scaling_factor, scaling_factor])

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target = 'Dragon'
torch.set_default_device('cpu')
torch.manual_seed(41)
model = INR3D(device='cpu')
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load(f'3d-model-{target}-2000', map_location='cpu'))

'''
x = torch.linspace(-0.5, 0.0, 100)
y = torch.linspace(-0.25, 0.25, 100)
z = torch.linspace(-0, 0.5, 100)
'''
x = torch.linspace(-0.5, 0.5, 100)
y = torch.linspace(-0.5, 0.5, 100)
z = torch.linspace(-0.5, 0.5, 100)
X, Y, Z = torch.meshgrid(x, y, z)
points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

with torch.no_grad():
    model(points[0])
    densities = np.array(model(points))

print(points.shape)
grid = points.reshape(100, 100, 100, 3)
print(grid.shape)
densities = densities.reshape(100, 100, 100)

plane = densities[:, :, 50]
fig = plt.figure()
plt.imshow(plane)
# plt.show(block=True)

vertices, faces, normals, values = measure.marching_cubes(densities, allow_degenerate=False, level=0)
vertices = vertices * ((np.array([0.5, 0.5, 0.5]) - np.array([-0.5, -0.5, -0.5])) / densities.shape) + np.array(
    [-0.5, -0.5, -0.5])

p1 = pv.Plotter()
p1.add_points(pv.PolyData(vertices))
p1.add_axes()
p1.show_grid()
# p1.show()

### Camera position
camera_position = [(100., 100., 55),
                   (-0.05642535239457658, -0.3681839525699573, 20.08591179996729),
                   (0.0, 0.0, 1.0)]

'''
'''
optimal_points = utils.find_optimal_point_parallel(model, vertices, normals, 0.0001,
                                                   30, False)
print(sum([o[0] for o in optimal_points]), len(optimal_points))

# rmse_error = utils.rmse_model_evaluation(model, mesh.points, mesh.active_normals, False, camera_position,
#                                         epsilon=0.0001)
# print(f"RMSE: {rmse_error[0]}")

optimal_points = [o[0] if o[1] == 0 else vertices[i] for i, o in enumerate(optimal_points)]

vertices = np.array(optimal_points)
pv_faces = []

for face in faces:
    complete_face = face.tolist()
    complete_face.insert(0, len(face))
    pv_faces.append(complete_face)

surf = pv.PolyData(vertices, np.hstack([pv_faces]))
# surf.save(f'paper/output-meshes/{target}.ply')

plotter = pv.Plotter()
plotter.add_mesh(surf, show_edges=False, color='lightblue')

# plotter.camera_position = camera_position
plotter.show_axes()
plotter.show_grid()
plotter.enable_eye_dome_lighting()
plotter.show()


def bilateral_mesh_smoothing(mesh, iterations=10, sigma_s=0.1, sigma_n=0.1):
    for _ in range(iterations):
        new_vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals

        for i, vertex in enumerate(mesh.vertices):
            neighbors = mesh.vertex_neighbors[i]
            if not neighbors:
                continue

            weights = []
            displacement = np.zeros(3)

            for j in neighbors:
                diff = mesh.vertices[j] - vertex
                spatial_weight = np.exp(-np.linalg.norm(diff) ** 2 / (2 * sigma_s ** 2))
                normal_weight = np.exp(-np.linalg.norm(normals[j] - normals[i]) ** 2 / (2 * sigma_n ** 2))
                weight = spatial_weight * normal_weight
                weights.append(weight)
                displacement += weight * diff

            if sum(weights) > 0:
                new_vertices[i] += displacement / sum(weights)

        mesh.vertices = new_vertices

    return mesh


plotter = pv.Plotter()

surf = trimesh.Trimesh(vertices=vertices,
                       faces=faces)

hej = bilateral_mesh_smoothing(surf)
print("hej")

plotter.add_mesh(hej, show_edges=False, color='lightblue')

# plotter.camera_position = camera_position
plotter.show_axes()
plotter.show_grid()
plotter.enable_eye_dome_lighting()
plotter.show()

# plotter.screenshot('test2.png', False)
# plotter.save_graphic('test2.svg', title='PyVista Export', raster=True, painter=True)
