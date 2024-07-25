import math
import random

import cv2 as cv
import numpy as np
import pyvista as pv
import torch
from matplotlib import pyplot as plt
from scipy import spatial
from sklearn.neighbors import KDTree

import gibbs
from inr_model import INR3D


def gear(angle):
    return 100 + (10 * math.sin(20 * math.radians(angle)))


def gear_custom(angle, a, b, c):
    return a + (b * math.sin(c * math.radians(angle)))


def oracle(point):
    radius, angle = convert_cartesian_to_polar((250, 250), point)
    diff = radius - gear(angle)
    return diff  # np.sign(diff)


def realistic_oracle(point):
    return np.sign(oracle(point))


def convert_cartesian_to_polar(center, point):
    vector = (point[0] - center[0], point[1] - center[1])
    radius = np.linalg.norm(vector)
    angle = math.atan2(vector[1], vector[0])

    angle_degrees = math.degrees(angle)
    if angle_degrees < 0:
        angle_degrees += 360

    return radius, angle_degrees


def convert_polar_to_cartesian(angle, radius, center):
    center_x, center_y = center
    return (radius * math.cos(math.radians(angle))) + center_x, (radius * math.sin(math.radians(angle))) + center_y


def fall_to_nearest_ray(point, center, ray_number):
    radius, angle = convert_cartesian_to_polar(center, point)
    angle = round(angle / ray_number) * ray_number
    return convert_polar_to_cartesian(angle, radius, center)


def simulate_laser_ray(start_point, angle, direction, frame):
    direction = 1
    cv.drawMarker(frame, start_point, (255, 255, 255), cv.MARKER_TRIANGLE_UP, 10, 1)
    p = np.array(np.around(convert_polar_to_cartesian(angle, -500 * direction, start_point)), dtype=int)
    p1 = np.array(np.around(convert_polar_to_cartesian(angle, 500 * direction, start_point)), dtype=int)
    cv.drawMarker(frame, p1, [255, 255, 255], cv.MARKER_DIAMOND, 10, 1)
    p2 = np.array(np.around(convert_polar_to_cartesian(angle, 50 * direction, start_point)), dtype=int)
    cv.arrowedLine(frame, start_point, p2, (100, 100, 100), 1, tipLength=0.5)

    has_collided = False

    for r in range(-500, 500):
        radius = r * direction
        test_point = np.array(np.around(convert_polar_to_cartesian(angle, radius, start_point)), dtype=int)
        if oracle(test_point) < 0:
            cv.line(frame, p, test_point, (255, 255, 255), 1)
            cv.line(frame, test_point, p1, (100, 100, 100), 1)
            has_collided = True
            break

    if not has_collided:
        cv.line(frame, p, p1, (255, 255, 255), 1)


def generate_laser_points(start_point, angle):
    external = []
    edge = []
    unknown = []

    found = False
    for radius in range(-500, 500):
        test_point = np.array(np.around(convert_polar_to_cartesian(angle, radius, start_point)), dtype=int)
        if oracle(test_point) > 0 and not found:
            external.append(test_point)
        elif -5 <= oracle(test_point) <= 0:
            edge.append(test_point)
        else:
            unknown.append(test_point)
            found = True

    return external, edge, unknown


def knn_point_classification(external, internal, unknown, k=5):
    ret_external = [] + external
    ret_internal = [] + internal
    all_points = [] + external + internal
    kd_tree = spatial.KDTree(all_points)
    _, neighbors = kd_tree.query(unknown, k=k)

    all_labels = [1 for _ in external] + [-1 for _ in internal]
    for i, u in enumerate(unknown):
        point_neighbors = neighbors[i]
        if k == 1:
            point_class = sum([all_labels[point_neighbors]])
        else:
            point_class = sum([all_labels[n] for n in point_neighbors])

        if point_class <= 0:
            ret_internal.append(u)
        else:
            ret_external.append(u)

    return ret_external, ret_internal


def pure_knn_point_classification(external, internal, unknown, k=5):
    ret_external = [] + external
    ret_internal = [] + internal
    all_points = [] + external + internal + unknown
    all_labels = [1 for _ in external] + [-1 for _ in internal] + [0 for _ in unknown]
    kd_tree = KDTree(np.array(all_points))
    distances, neighbors = kd_tree.query(unknown, k=k, return_distance=True)

    for i, u in enumerate(unknown):
        point_neighbors = neighbors[i]
        if k == 1:
            point_class = [all_labels[point_neighbors]]
            distances_sum = distances[i]
        else:
            point_class = [all_labels[n] for n in point_neighbors]
            distances_sum = sum(distances[i])

        if sum(distances[i]) == 0:
            external_score = len(
                [p for index, p in enumerate(point_class) if p == 1])
            unknown_score = len(
                [p for index, p in enumerate(point_class) if p == 0])
            internal_score = len(
                [p for index, p in enumerate(point_class) if p == -1])
        else:
            external_score = len(
                [(1 - (distances[index] / distances_sum)) * p for index, p in enumerate(point_class) if p == 1])
            unknown_score = len(
                [(1 - (distances[index] / distances_sum)) * p for index, p in enumerate(point_class) if p == 0])
            internal_score = len(
                [(1 - (distances[index] / distances_sum)) * p for index, p in enumerate(point_class) if p == -1])

        if unknown_score > external_score:
            ret_internal.append(u)
        else:
            ret_external.append(u)
        '''
        if external_score >= internal_score and external_score >= unknown_score >= internal_score:
            ret_external.append(u)
        else:
            ret_internal.append(u)
        '''

        '''
        if external_score > unknown_score or external_score > internal_score:
            ret_external.append(u)
        else:
            ret_internal.append(u)
        '''

    return ret_external, ret_internal


def project_point_onto_vector(P, A, v):
    P = np.array(P)
    A = np.array(A)
    v = np.array(v)

    P_translated = P - A
    dot_product = np.dot(P_translated, v)
    vector_magnitude_squared = np.dot(v, v)
    projection_scalar = dot_product / vector_magnitude_squared
    projection = projection_scalar * v
    projected_point = projection + A

    return projected_point


def determine_side_of_vector(P, A, v):
    projected_point = project_point_onto_vector(P, A, v)
    AP_projected = projected_point - A
    dot_product = np.dot(AP_projected, v)
    if dot_product > 0:
        return 1
    elif dot_product < 0:
        return -1
    else:
        return 0


def normal_based_knn_point_classification(vertex, normals, unknown, k=5):
    ret_external = []
    ret_internal = []
    kd_tree = spatial.KDTree(vertex)
    _, neighbors = kd_tree.query(unknown, k=k)

    for i, u in enumerate(unknown):
        point_neighbors = neighbors[i]
        if k == 1:
            point_class = sum([[determine_side_of_vector(u, vertex[point_neighbors], normals[point_neighbors])]])
        else:
            point_class = sum([determine_side_of_vector(u, vertex[n], normals[n]) for n in point_neighbors])

        if point_class <= 0:
            ret_internal.append(u)
        else:
            ret_external.append(u)

    return ret_external, ret_internal


def is_point_inside_mesh(mesh, point):
    inside = mesh.select_enclosed_points(pv.PolyData([point]), tolerance=0.0, inside_out=True)
    return inside['SelectedPoints'][0] == 1


def rotate_x(angle):
    return np.array([
        [1, 0, 0],
        [0, math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
        [0, math.sin(math.radians(angle)), math.cos(math.radians(angle))]
    ])


def rotate_y(angle):
    return np.array([
        [math.cos(math.radians(angle)), 0, math.sin(math.radians(angle))],
        [0, 1, 0],
        [-math.sin(math.radians(angle)), 0, math.cos(math.radians(angle))]
    ])


def rotate_z(angle):
    return np.array([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
        [0, 0, 1]
    ])


def find_plane_line_intersection(plane, point1, point2):
    """
    Find intersection between a plane and the line passing through two points
    :param plane:
    :param point1:
    :param point2:
    :return:
    """
    direction = point2 - point1
    plane_norm = np.array([plane[0], plane[1], plane[2]])
    product = plane_norm @ direction
    if abs(product) > 1e-6:
        p_co = plane_norm * (-plane[3] / (plane_norm @ plane_norm))

        w = point1 - p_co
        fac = - (plane_norm @ w) / product
        return point1 + (direction * fac)

    return None


def find_line_equation(x1, y1, x2, y2):
    """
    Find the equation of the line passing through two points
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: the line params in the form a, b, c
    """
    if x2 - x1 == 0:
        a = 1
        b = 0
        c = -x1
    else:
        m = (y2 - y1) / (x2 - x1)
        a = -m
        b = 1
        c = m * x1 - y1

    return a, b, c


def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx / 2
    y_step = 1 if y0 < y1 else -1
    y = y0

    for x in range(x0, x1 + 1):
        if steep:
            yield y, x
        else:
            yield x, y

        error -= dy
        if error < 0:
            y += y_step
            error += dx


def project_point(point, rotation_matrix, translation_vector, camera_intrinsic_matrix):
    point.append(1)
    camera_p = camera_intrinsic_matrix @ np.concatenate([rotation_matrix, np.matrix(translation_vector).T],
                                                        axis=1) @ point
    return [int(round(camera_p[0, 0] / camera_p[0, 2])), int(round(camera_p[0, 1] / camera_p[0, 2]))]


def cross_product_proxy(a, b):
    return np.cross(a, b)


def compute_laser_transformation(laser_center, laser_norm):
    translation_vector = np.array(laser_center)
    laser_norm /= np.linalg.norm(laser_norm)

    if np.dot(laser_norm, np.array([1, 0, 0])) < 0:
        laser_norm = -laser_norm

    u = np.array([-laser_norm[1], laser_norm[0], 0]) / np.linalg.norm(np.array([-laser_norm[1], laser_norm[0], 0]))
    v = (cross_product_proxy(laser_norm, u)) / np.linalg.norm(cross_product_proxy(laser_norm, u))

    R = np.column_stack((u, v, laser_norm))

    return R, translation_vector


def sample_point_from_plane(laser_center, laser_norm):
    R, t = compute_laser_transformation(laser_center, laser_norm)

    x = random.uniform(-2, 2)
    y = random.uniform(-6, 6)

    z = 0

    point = np.array([x, y, z, 1])
    world_point = np.concatenate(
        [np.concatenate([R, np.matrix(t).T], axis=1),
         np.array([[0, 0, 0, 1]])], axis=0) @ point

    world_point = np.squeeze(np.asarray(world_point))
    return [world_point[0] / world_point[3], world_point[1] / world_point[3], world_point[2] / world_point[3]]


def inverse_cdf(p, x, cdf):
    index = np.searchsorted(cdf, p)
    if index == len(x):
        index -= 1
    return x[index]


def closest(lst, k):
    return (torch.abs(lst - k)).argmin()


def autograd_proxy(output, input_tensor):
    grad_outputs = torch.autograd.grad(outputs=output, inputs=input_tensor, grad_outputs=torch.ones_like(output),
                                       is_grads_batched=False)[0]

    gradient_image = torch.tensor([math.sqrt((x ** 2) + (y ** 2) + (z ** 2)) for [x, y, z] in grad_outputs],
                                  dtype=torch.float32)
    return gradient_image.cpu()


def sample_point_from_plane_gradient(laser_center, laser_norm, model, k=100):
    points = []
    R, t = compute_laser_transformation(laser_center, laser_norm)
    x = torch.linspace(-20, 20, 50, device='cpu')
    y = torch.linspace(-60, 60, 100, device='cpu')
    z = 0

    grid_points = []
    for i in x:
        for j in y:
            point = np.array([i, j, z, 1])
            world_point = np.concatenate(
                [np.concatenate([R, np.matrix(t).T], axis=1),
                 np.array([[0, 0, 0, 1]])], axis=0) @ point

            world_point = np.squeeze(np.asarray(world_point))
            grid_points.append(
                [world_point[0] / world_point[3], world_point[1] / world_point[3], world_point[2] / world_point[3]])

    grid_points = torch.tensor(grid_points, dtype=torch.float32)

    grid_points_clone = grid_points.clone()
    # grid_points_clone /= 10
    grid_points = grid_points.requires_grad_(True)
    output = model(grid_points)
    gradient_image = autograd_proxy(output, grid_points)

    output = output.view(50, 100)
    output = output.detach().cpu().numpy()
    dbg = False
    if dbg:
        plane = output
        fig = plt.figure()
        plt.imshow(plane)
        # plt.show(block=True)
        plt.show(block=False)

        plane = gradient_image.view(50, 100)
        fig = plt.figure()
        plt.imshow(plane)
        # plt.show(block=True)
        plt.show(block=False)

    '''
    x_distr, _ = margins(gradient_image.view(50, 100).numpy())

    for _ in range(k):
        x = x_distr.flatten()
        x /= np.sum(x)
        cdf = np.cumsum(x)
        probabilities_to_invert = np.random.uniform(0, 1, 1)
        # print(np.searchsorted(cdf, 0))
        # print(np.searchsorted(cdf, 1))
        # FIXME: this might be broken
        x = closest(torch.linspace(-20, 20, 50),
                    [inverse_cdf(p, torch.linspace(-20, 20, 50), cdf) for p in probabilities_to_invert][0])

        y_distr = margins(gradient_image.view(50, 100)[x, :].numpy())[0]

        y = y_distr.flatten()
        y /= np.sum(y)
        cdf = np.cumsum(y)
        probabilities_to_invert = np.random.uniform(0, 1, 1)
        y = closest(torch.linspace(-40, 40, 100),
                    [inverse_cdf(p, torch.linspace(-40, 40, 100), cdf) for p in probabilities_to_invert][0])

        x_distr = margins(gradient_image.view(50, 100)[:, y].numpy())[0]

        points.append(grid_points_clone.view(50, 100, 3)[x, y].cpu().numpy())
    '''

    points = gibbs.gibbs_sampling_2d(gradient_image.view(50, 100).detach().cpu().numpy(), k, [0, 0, 0],
                                     grid_points_clone)

    return torch.from_numpy(np.array(points)), grid_points_clone


def abs_model_evaluation(model: INR3D, points: np.ndarray):
    with torch.no_grad():
        densities = model(torch.tensor(points, dtype=torch.float32))
        outputs = abs(densities.detach().cpu().numpy())
    return np.sum(outputs) / len(points)


def find_optimal_point_parallel(model, vertexes, normals, epsilon=1e-5, max_iter=100, dbg=False):
    a = np.full(len(vertexes), -0.2, dtype=np.float32)
    b = np.full(len(vertexes), 0.2, dtype=np.float32)
    num_points = len(vertexes)

    optimal_points = [[vertexes[i], 1] for i in range(len(vertexes))]

    for _ in range(max_iter):
        m = (a + b) / 2.0

        points_low = vertexes + a[:, np.newaxis] * normals
        points_mid = vertexes + m[:, np.newaxis] * normals
        points_high = vertexes + b[:, np.newaxis] * normals

        points = np.vstack((points_low, points_mid, points_high))
        values = model(torch.from_numpy(points).type(torch.float32).to(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )).detach().cpu()

        val_a = values[:num_points]
        val_m = values[num_points:2 * num_points]
        val_b = values[2 * num_points:]

        mask = np.abs(val_m) < epsilon
        for i, flag in enumerate(mask):
            if flag and optimal_points[i][1] == 1:
                optimal_points[i] = [points_mid[i], 0]

        sign_low = np.sign(val_a).flatten()
        sign_mid = np.sign(val_m).flatten()
        sign_high = np.sign(val_b).flatten()

        for i in range(len(a)):
            if sign_low[i] != sign_mid[i]:
                b[i] = m[i]
            elif sign_high[i] != sign_mid[i]:
                a[i] = m[i]
            else:
                a[i] *= 1.2
                b[i] *= 1.2

        if dbg:
            plotter = pv.Plotter()
            plotter.add_points(pv.PolyData(points_low), color='red')
            plotter.add_points(pv.PolyData(points_high), color='blue')
            plotter.add_points(pv.PolyData(points_mid), color='green')
            plotter.show()

    return optimal_points


def mae_model_evaluation(model: INR3D, points: np.ndarray, normals: np.ndarray):
    abs_values = []
    optimal_points = find_optimal_point_parallel(model, points,
                                                 normals,
                                                 epsilon=0.0001,
                                                 max_iter=30,
                                                 dbg=False)
    print("converged on mesh: ", sum([1 for o in optimal_points if o[1] == 0]), len(points))
    errors = []
    for i in range(len(points)):
        point = points[i]
        if optimal_points[i][1] == 0:
            optimal_point = optimal_points[i][0]
            # print(math.dist(point, optimal_point))
            error = math.dist(point, optimal_point)
            # if error > 5:
            #    error = 5
            abs_values.append(error)
            errors.append(error)
        else:
            errors.append(np.nan)
            # squares.append(1 ** 2)

    return sum(abs_values) / len(points), sum([1 for o in optimal_points if o[1] == 0]) / len(points)


def rmse_model_evaluation(model: INR3D, points: np.ndarray, normals: np.ndarray, dbg=False):
    squares = []
    optimal_points = find_optimal_point_parallel(model, points,
                                                 normals,
                                                 epsilon=0.0001,
                                                 max_iter=30,
                                                 dbg=False)
    print("converged on mesh: ", sum([1 for o in optimal_points if o[1] == 0]), len(points))
    errors = []
    for i in range(len(points)):
        point = points[i]
        if optimal_points[i][1] == 0:
            optimal_point = optimal_points[i][0]
            error = math.dist(point, optimal_point)
            # if error > 5:
            #    error = 5
            squares.append(error ** 2)
            errors.append(error)
        else:
            errors.append(np.nan)
    '''
    p1 = pv.Plotter()
    p1.add_points(pv.PolyData(optimal_points), color='red')
    p1.add_points(pv.PolyData(points), color='blue')
    # p1.add_points(pv.PolyData(old_vertices))
    # p1.add_arrows(points, normals, color='black')
    p1.add_axes()
    p1.show_grid()
    p1.show()
    '''

    if dbg:
        point_cloud = pv.PolyData(points)
        point_cloud['errors'] = [min(e, 2.) for e in errors]
        cmap = plt.cm.plasma
        cmap.set_bad(color='green')
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud, scalars='errors', cmap=cmap, nan_color='green', point_size=10)
        plotter.add_scalar_bar(title='Error Value', n_labels=5)

        plotter.show()

    return math.sqrt(sum(squares) / len(points)), np.array(errors), \
        sum([1 for o in optimal_points if o[1] == 0]) / len(points)
