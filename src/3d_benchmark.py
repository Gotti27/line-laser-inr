import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from inr_model import INR3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
model = INR3D()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('3d-model', map_location=device))

x = torch.linspace(-30, 30, 100)
y = torch.linspace(-30, 0, 50)
z = torch.linspace(-30, 30, 100)
X, Y, Z = torch.meshgrid(x, y, z)

points = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

with torch.no_grad():
    densities = np.array(model(points).reshape(100, 50, 100))

verts, faces, normals, values = measure.marching_cubes(densities)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection(verts[faces])
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
