import os

import numpy as np
import pyvista
import pyvista as pv
import torch
import trimesh
from skimage import measure

import utils
from dataset import load_renders
from inr_model import INR3D
from src.evaluation import metrics
from src.utils import rotate_z

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def silhouette_sampling(point):
    a, b, c = point
    # for image in list(filter(lambda img: 'right' in img, images)):
    for image in images:
        K = renders_matrices[image]['K']
        R = renders_matrices[image]['R']
        t = renders_matrices[image]['t']
        render_depth = renders_matrices[image]['render']

        p = utils.project_point([a, b, c], R, t, K)
        depth = render_depth[:, :, 3]

        is_outside = p[0] < 0 or p[0] >= 256 or p[1] < 0 or p[1] >= 256
        if not is_outside and depth[p[1], p[0]] == 0:
            return 1
    return -1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target = 'bunny-small'
mode = 'uniform'
post_process = False
torch.manual_seed(41)
model = INR3D()

image_folder = f'renders/{target}'
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images.sort(key=lambda name: int(name.split('_')[1]))
renders_matrices = load_renders(images, target)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(
    torch.load('models/20250506_041016_bunny-small_gradient/3d-model-bunny-small-60',
               map_location=device))

x = torch.linspace(-0.5, 0.5, 100)
y = torch.linspace(-0.5, 0.5, 100)
z = torch.linspace(-0.5, 0.5, 100)
X, Y, Z = torch.meshgrid(x, y, z)
points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

with torch.no_grad():
    '''
    densities = np.empty((0, 1))
    for chunk in points.chunk(1000):
        print(np.array(model(chunk)).shape)
        densities = np.concatenate([densities,
                                    np.array(model(chunk))])  # np.array(model(points))  # .reshape(100, 50, 100))
    '''
    densities = np.array(model(points))

print(points.shape)
grid = points.reshape(100, 100, 100, 3)
print(grid.shape)

'''
'''

# PostProcessing routine
'''

def just_tell_me_where_are_you(i):
    print(f"{i * 100 / len(points)}", end='\r')
    return True

'''
'''
'''
if post_process:
    densities = [1 if densities[i] < 0 and (silhouette_sampling(points[i] / 10)) == 1 else densities[i][0] for i in
                 range(len(points))]

densities = np.array(densities)
densities = densities.reshape(100, 100, 100)

'''
for i in range(200):
    for j in range(100):
        for k in range(200):
            if densities[i, j, k] < 0 and (silhouette_sampling(grid[i, j, k] / 10)) == 1:
                densities[i, j, k] = 1
    print(f"row: {i} done")
'''
'''
'''
#

'''
plane = densities[:, :, 100]
fig = plt.figure()
plt.imshow(plane)
plt.show(block=True)
'''

vertices, faces, normals, values = measure.marching_cubes(densities, allow_degenerate=False, level=0)
vertices = vertices * ((np.array([0.5, 0.5, 0.5]) - np.array([-0.5, -0.5, -0.5])) / densities.shape) + np.array(
    [-0.5, -0.5, -0.5])

'''
old_vertices = copy.deepcopy(vertices)
point_cloud = pv.PolyData(vertices)
point_cloud.plot(eye_dome_lighting=True, show_axes=True, show_grid=True)
'''

mesh = pyvista.read(f'scenes/meshes/{target}.ply')
mesh.compute_normals(inplace=True)
# mesh = mesh.rotate_z(180)
# mesh = mesh.rotate_x(90)
# mesh = mesh.scale(10)
# print([model(torch.tensor([p])) for p in mesh.points])
# mesh = mesh.translate()
p1 = pv.Plotter()
# p1.add_points(mesh, color='tan')
# p1.add_points(pv.PolyData(vertices))
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
p1.add_mesh(surf)
# p1.add_points(pv.PolyData(points.detach().numpy()), color='red')
# p1.add_arrows(mesh.points, mesh.active_normals, color='black')
p1.add_axes()
p1.show_grid()
p1.show()

'''
abs_error = utils.abs_model_evaluation(model, mesh.points)
print(f"ABS: {abs_error}")
mae_error = utils.mae_model_evaluation(model, mesh.points, mesh.active_normals)
print(f"MAE: {mae_error[0]}")
'''

mesh_points_indexes = [i for i in range(len(mesh.points)) if True]  # mesh.points[i][1] < -1]

### Camera position

'''
camera_position = [(100., 100., 55),
                   (-0.05642535239457658, -0.3681839525699573, 20.08591179996729),
                   (0.0, 0.0, 1.0)]
'''

###

'''
rmse_error = utils.rmse_model_evaluation(model, mesh.points[mesh_points_indexes],
                                         mesh.active_normals[mesh_points_indexes], True, camera_position)
print(f"RMSE: {rmse_error[0]}")
'''

# optimal_points = [utils.find_optimal_point(model, vertices[i], normals[i], epsilon=0.001) for i in range(len(vertices))]
optimal_points = utils.find_optimal_point_parallel(model, vertices, normals, 0.0001, 30, False)
print(sum([o[1] for o in optimal_points]), len(mesh_points_indexes))

optimal_points = [o[0] if o[1] == 0 else vertices[i] for i, o in enumerate(optimal_points)]
# optimal_points = [o[0] if o[1] == 0 else mesh.points[i] for i, o in enumerate(optimal_points)]

# point_cloud = pv.PolyData(optimal_points)
# point_cloud.plot(eye_dome_lighting=True, border_color='green')

vertices = np.array(optimal_points)
# vertices = np.array(vertices)

# cloud = pv.PolyData(vertices)
# cloud.plot()

pv_faces = []

for face in faces:
    complete_face = face.tolist()
    complete_face.insert(0, len(face))
    pv_faces.append(complete_face)

surf = pv.PolyData(vertices, np.hstack([pv_faces]))
surf.save(f'models/output-meshes/{target}/{mode}-smooth.ply')

plotter = pv.Plotter()
plotter.add_mesh(surf, show_edges=False, color='lightblue')
'''
plotter.view_isometric()
camera_position = plotter.camera_position
camera_location, camera_focus, view_up = camera_position
lowered_camera_location = (camera_location[0] + 50, camera_location[1], camera_location[2] - 100)
plotter.camera_position = (lowered_camera_location, camera_focus, view_up)
print(plotter.camera_position)
'''
# plotter.camera_position = camera_position
plotter.show_axes()
# plotter.show_grid()
# plotter.enable_eye_dome_lighting()

plotter.show()
plotter.screenshot('test2.png', False)
# plotter.save_graphic('test2.svg', title='PyVista Export', raster=True, painter=True)


vertices = surf.points
faces = surf.faces.reshape((-1, 4))[:, 1:]

num_points = 100000
points, _ = trimesh.sample.sample_surface(
    trimesh.Trimesh(vertices=vertices, faces=faces)
    , num_points)

gt_points, _ = trimesh.sample.sample_surface(
    trimesh.load(f'scenes/meshes/{target}.ply')
    , num_points)

gt_points @= rotate_z(180)

pl = pv.Plotter()
pl.add_points(points)
pl.add_points(gt_points, color='orange')
pl.show_grid()
pl.show_axes()
pl.enable_eye_dome_lighting()
pl.show()

print(metrics.chamfer_distance(points, gt_points))
print(metrics.hausdorff_distance(points, gt_points))
