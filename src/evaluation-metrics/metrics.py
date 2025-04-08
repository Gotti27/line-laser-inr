import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(pc1, pc2):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)

    return np.mean(dist1 ** 2) + np.mean(dist2 ** 2)
