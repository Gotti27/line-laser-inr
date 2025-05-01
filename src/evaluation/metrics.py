import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(pc1, pc2):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)

    return np.mean(dist1 ** 2) + np.mean(dist2 ** 2)


def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    def one_sided_hausdorff(a, b):
        tree_b = cKDTree(b)
        distances, _ = tree_b.query(a, k=1)
        return np.max(distances)

    h_ab = one_sided_hausdorff(pc1, pc2)
    h_ba = one_sided_hausdorff(pc2, pc1)
    return max(h_ab, h_ba)
