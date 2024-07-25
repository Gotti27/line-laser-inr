import copy

import matplotlib.pyplot as plt
import numpy as np
import pyvista
import pyvista as pv
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import utils
from inr_model import INR3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target = 'Dragon'
torch.manual_seed(41)
model = INR3D()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('3d-model', map_location=device))

'''
with open('history-uniform.txt', 'r') as f:
    history_uniform = []
    for line in f.readlines():
        history_uniform.append(float(line.strip().split()[0]))

with open('history-gradient.txt', 'r') as f:
    history_gradient = []
    for line in f.readlines():
        history_gradient.append(float(line.strip().split()[0]))

plt.plot(history_uniform, label='uniform')
plt.plot(history_gradient, label='gradient')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Root Mean Square Error")
plt.show()
'''

x = torch.linspace(-40, 40, 200)
y = torch.linspace(-40, 0, 100)
z = torch.linspace(-40, 40, 200)
X, Y, Z = torch.meshgrid(x, y, z)

points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

with torch.no_grad():
    densities = np.array(model(points).reshape(200, 100, 200))

plane = densities[:, :, 100]
fig = plt.figure()
plt.imshow(plane)
plt.show(block=True)

vertices, faces, normals, values = measure.marching_cubes(densities, allow_degenerate=False, level=0)

vertices = vertices * ((np.array([40, 0, 40]) - np.array([-40, -40, -40])) / densities.shape) + np.array(
    [-40, -40, -40])

old_vertices = copy.deepcopy(vertices)
point_cloud = pv.PolyData(vertices)
point_cloud.plot(eye_dome_lighting=True, show_axes=True, show_grid=True)

mesh = pyvista.read(f'scenes/meshes/{target}.ply')
mesh.compute_normals(inplace=True)
mesh = mesh.rotate_z(180)
mesh = mesh.rotate_x(90)
mesh = mesh.scale(10)
# print([model(torch.tensor([p])) for p in mesh.points])
# mesh = mesh.translate()
p1 = pv.Plotter()
p1.add_points(mesh, color='tan')
p1.add_points(pv.PolyData(old_vertices))
# p1.add_points(pv.PolyData(points.detach().numpy()), color='red')
p1.add_arrows(mesh.points, mesh.active_normals, color='black')
p1.add_axes()
p1.show_grid()
p1.show()

abs_error = utils.abs_model_evaluation(model, mesh.points)
print(f"ABS: {abs_error}")
mae_error = utils.mae_model_evaluation(model, mesh.points, mesh.active_normals)
print(f"MAE: {mae_error[0]}")

mesh_points_indexes = [i for i in range(len(mesh.points)) if mesh.points[i][1] < -1]

rmse_error = utils.rmse_model_evaluation(model, mesh.points[mesh_points_indexes],
                                         mesh.active_normals[mesh_points_indexes], True)
print(f"RMSE: {rmse_error[0]}")

# optimal_points = [utils.find_optimal_point(model, vertices[i], normals[i], epsilon=0.001) for i in range(len(vertices))]
optimal_points = utils.find_optimal_point_parallel(model, vertices, normals, 0.0001, 30, False)
print(sum([o[1] for o in optimal_points]), len(mesh_points_indexes))
optimal_points = [o[0] if o[1] == 0 else vertices[i] for i, o in enumerate(optimal_points)]
'''
for _ in range(100):
    nd_vertices = np.array(vertices)

    with torch.no_grad():
        positive = [v + (coeff * normals[i]) for i, v in enumerate(nd_vertices)]
        negative = [v + (-coeff * normals[i]) for i, v in enumerate(nd_vertices)]
        pos_densities = model(torch.tensor(positive, dtype=torch.float32))
        densities = model(torch.tensor(nd_vertices, dtype=torch.float32))
        neg_densities = model(torch.tensor(negative, dtype=torch.float32))
        updated_vertices = []
        for i, vertice in enumerate(vertices):
            if abs(densities[i]) < epsilon:
                updated_vertices.append(vertice)
            elif math.copysign(1, neg_densities[i]) == math.copysign(1, densities[i]):
                updated_vertices.append((positive[i] + vertice) / 2)
            else:
                updated_vertices.append((negative[i] + vertice) / 2)

        vertices = updated_vertices
        coeff /= 2
'''
point_cloud = pv.PolyData(optimal_points)
point_cloud.plot(eye_dome_lighting=True, border_color='green')

# vertices = [v.tolist() for v in vertices]
# vertices = np.array(vertices)
vertices = np.array(optimal_points)

cloud = pv.PolyData(vertices)
cloud.plot()

volume = cloud.delaunay_3d(alpha=1)  # , progress_bar=True)  #
shell = volume.extract_geometry()
shell.plot(eye_dome_lighting=True, show_axes=True)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# mesh = Poly3DCollection(old_vertices[faces])
mesh = Poly3DCollection(np.array([[vertices[f] for f in face] for face in faces]))
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)

plt.tight_layout()
plt.show()
