import math
import random

import cv2 as cv
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import spatial
from scipy.stats.contingency import margins


def gear(angle):
    return 100 + (10 * math.sin(20 * math.radians(angle)))


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
    # in pure knn internal list should be empty
    ret_external = [] + external
    ret_internal = [] + internal
    all_points = [] + external + internal + unknown
    kd_tree = spatial.KDTree(all_points)
    _, neighbors = kd_tree.query(unknown, k=k)

    all_labels = [1 for _ in external] + [-1 for _ in internal] + [0 for _ in unknown]
    for i, u in enumerate(unknown):
        point_neighbors = neighbors[i]
        if k == 1:
            point_class = [all_labels[point_neighbors]]
        else:
            point_class = [all_labels[n] for n in point_neighbors]

        internal_score = len([p for p in point_class if p == -1])
        external_score = len([p for p in point_class if p == 1])
        unknown_score = len([p for p in point_class if p == 0])

        # FIXME: this condition has to be refactored
        if unknown_score > external_score or internal_score > unknown_score:
            ret_internal.append(u)
        else:
            ret_external.append(u)

    return ret_external, ret_internal


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


def sample_point_from_plane(plane, degree_threshold, side):
    a, b, c, d = plane

    if side == 'right':
        y = random.uniform(-3, 0)
    else:
        y = random.uniform(-3, 0)  # TODO: update this to use different ranges given the side

    if 45 <= degree_threshold < 135:
        x = random.uniform(-3, 3)
        z = -(a * x + b * y + d) / c
    elif 135 <= degree_threshold < 225:
        z = random.uniform(-3, 3)
        x = -(c * z + b * y + d) / a
    elif 225 <= degree_threshold < 315:
        x = random.uniform(-3, 3)
        z = -(a * x + b * y + d) / c
    else:
        z = random.uniform(-3, 3)
        x = -(c * z + b * y + d) / a

    return [x, y, z]


def inverse_cdf(p, x, cdf):
    return x[np.searchsorted(cdf, p)]


def closest(lst, k):
    return (torch.abs(lst - k)).argmin()


def sample_point_from_plane_gradient(plane, degree_threshold, side, gradient_image, k=100):
    a, b, c, d = plane

    points = []
    _, y_distribution, _ = margins(gradient_image[:, :, :])

    for _ in range(k):
        y = y_distribution.flatten()
        y /= np.sum(y)
        cdf = np.cumsum(y)

        probabilities_to_invert = np.random.uniform(0, 1, 1)
        y = closest(torch.linspace(-30, 0, 50),
                    [inverse_cdf(p, torch.linspace(-30, 0, 50), cdf) for p in probabilities_to_invert][0])

        if 45 <= degree_threshold < 135:
            x_distribution, _ = margins(gradient_image[:, y, :])
            x = x_distribution.flatten()
            x /= np.sum(x)
            cdf = np.cumsum(x)
            probabilities_to_invert = np.random.uniform(0, 1, 1)
            x = closest(torch.linspace(-30, 30, 100),
                        [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

            z = -(a * torch.linspace(-30, 30, 100)[x] + b * torch.linspace(-30, 0, 50)[y] + d) / c
            points.append(
                [torch.linspace(-30, 30, 100)[x].item(), torch.linspace(-30, 0, 50)[y].item(), z.item()])

            y_distribution, _ = margins(gradient_image[x, :, :])

        elif 135 <= degree_threshold < 225:
            _, z_distribution = margins(gradient_image[:, y, :])
            z = z_distribution.flatten()
            z /= np.sum(z)
            cdf = np.cumsum(z)
            probabilities_to_invert = np.random.uniform(0, 1, 1)
            z = closest(torch.linspace(-30, 30, 100),
                        [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

            x = -(b * torch.linspace(-30, 0, 50)[y] + c * torch.linspace(-30, 30, 100)[z] + d) / a
            points.append(
                [x.item(), torch.linspace(-30, 0, 50)[y].item(), torch.linspace(-30, 30, 100)[z].item()])

            _, y_distribution = margins(gradient_image[:, :, z])

        elif 225 <= degree_threshold < 315:
            x_distribution, _ = margins(gradient_image[:, y, :])
            x = x_distribution.flatten()
            x /= np.sum(x)
            cdf = np.cumsum(x)
            probabilities_to_invert = np.random.uniform(0, 1, 1)
            x = closest(torch.linspace(-30, 30, 100),
                        [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

            z = -(a * torch.linspace(-30, 30, 100)[x] + b * torch.linspace(-30, 0, 50)[y] + d) / c
            points.append(
                [torch.linspace(-30, 30, 100)[x].item(), torch.linspace(-30, 0, 50)[y].item(), z.item()])

            y_distribution, _ = margins(gradient_image[x, :, :])
        else:
            _, z_distribution = margins(gradient_image[:, y, :])
            z = z_distribution.flatten()
            z /= np.sum(z)
            cdf = np.cumsum(z)
            probabilities_to_invert = np.random.uniform(0, 1, 1)
            z = closest(torch.linspace(-30, 30, 100),
                        [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

            x = -(b * torch.linspace(-30, 0, 50)[y] + c * torch.linspace(-30, 30, 100)[z] + d) / a
            points.append(
                [x.item(), torch.linspace(-30, 0, 50)[y].item(), torch.linspace(-30, 30, 100)[z].item()])

            _, y_distribution = margins(gradient_image[:, :, z])

    return [[p_p / 10 for p_p in p] for p in points]


def autograd_proxy(output, input_tensor):
    grad_outputs = torch.autograd.grad(outputs=output, inputs=input_tensor, grad_outputs=torch.ones_like(output),
                                       is_grads_batched=False)[0]
    # output.backward()
    # print("mid-1")

    if grad_outputs.device != 'cpu':
        grad_outputs.detach().cpu()

    gradient_image = torch.tensor([math.sqrt((x ** 2) + (y ** 2) + (z ** 2)) for [x, y, z] in grad_outputs],
                                  dtype=torch.float32, device='cpu')
    return gradient_image


def sample_point_from_plane_gradient_refactored(plane, degree_threshold, side, gradient_image, model, k=100):
    points = []
    a, b, c, d = plane

    y = torch.linspace(-30, 0, 50)

    if 45 <= degree_threshold < 135:
        x = torch.linspace(-30, 30, 100)
        # zz, yy = torch.meshgrid(z, y, indexing='ij')

        grid_points = []
        for i in x:
            for j in y:
                z = -(a * i + b * j + d) / c
                # print(x)
                grid_points.append([i, j, z])

        # zz = -(a * xx + b * yy + d) / c
        # grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = torch.tensor(grid_points, dtype=torch.float32)

        # xx = -(b * yy + c ** zz + d) / a

        # grid_points = torch.stack([xx, yy, zz], dim=-1)
    elif 135 <= degree_threshold < 225:
        z = torch.linspace(-30, 30, 100)
        # zz, yy = torch.meshgrid(z, y, indexing='ij')

        grid_points = []
        for i in z:
            for j in y:
                x = -(c * i + b * j + d) / a
                # print(x)
                grid_points.append([x, j, i])

        # zz = -(a * xx + b * yy + d) / c
        # grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = torch.tensor(grid_points, dtype=torch.float32)

        # xx = -(b * yy + c ** zz + d) / a

        # grid_points = torch.stack([xx, yy, zz], dim=-1)
    elif 225 <= degree_threshold < 315:
        x = torch.linspace(-30, 30, 100)
        # zz, yy = torch.meshgrid(z, y, indexing='ij')

        grid_points = []
        for i in x:
            for j in y:
                z = -(a * i + b * j + d) / c
                # print(x)
                grid_points.append([i, j, z])

        # zz = -(a * xx + b * yy + d) / c
        # grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = torch.tensor(grid_points, dtype=torch.float32)

        # xx = -(b * yy + c ** zz + d) / a

        # grid_points = torch.stack([xx, yy, zz], dim=-1)
    else:
        z = torch.linspace(-30, 30, 100)
        # zz, yy = torch.meshgrid(z, y, indexing='ij')

        grid_points = []
        for i in z:
            for j in y:
                x = -(c * i + b * j + d) / a
                # print(x)
                grid_points.append([x, j, i])

        # zz = -(a * xx + b * yy + d) / c
        # grid_points = torch.stack([xx, yy, zz], dim=-1)
        grid_points = torch.tensor(grid_points, dtype=torch.float32)

        # xx = -(b * yy + c ** zz + d) / a

        # grid_points = torch.stack([xx, yy, zz], dim=-1)

    grid_points_clone = grid_points.clone()
    grid_points_clone /= 10
    # grid_points = [[p_p / 10 for p_p in p] for p in grid_points]
    # grid_points = grid_points.flatten(start_dim=0, end_dim=1)
    # print("start-1")
    input_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=True)
    output = model(input_tensor)  # input_tensor
    gradient_image = autograd_proxy(output, input_tensor)

    # output = output.view(50, 100)
    output = output.view(100, 50)
    output = output.detach().cpu().numpy()
    dbg = False
    if dbg:
        plane = output
        fig = plt.figure()
        plt.imshow(plane)
        # plt.show(block=True)
        plt.show(block=False)

        plane = gradient_image.view(100, 50)
        fig = plt.figure()
        plt.imshow(plane)
        # plt.show(block=True)
        plt.show(block=False)

    x_distr, _ = margins(gradient_image.view(100, 50).numpy())

    for _ in range(k):
        x = x_distr.flatten()
        x /= np.sum(x)
        cdf = np.cumsum(x)
        probabilities_to_invert = np.random.uniform(0, 1, 1)
        x = closest(torch.linspace(-30, 30, 100),
                    [inverse_cdf(p, torch.linspace(-30, 30, 100), cdf) for p in probabilities_to_invert][0])

        '''
        plt.figure()
        plt.plot(x)
        plt.show(block=False)

        plt.figure()
        plt.plot(cdf)
        plt.show(block=False)
        '''

        y_distr = margins(gradient_image.view(100, 50)[x, :].numpy())[0]

        y = y_distr.flatten()
        y /= np.sum(y)
        cdf = np.cumsum(y)
        probabilities_to_invert = np.random.uniform(0, 1, 1)
        y = closest(torch.linspace(-30, 0, 50),
                    [inverse_cdf(p, torch.linspace(-30, 0, 50), cdf) for p in probabilities_to_invert][0])

        '''
        plt.figure()
        plt.plot(y)
        plt.show(block=False)

        plt.figure()
        plt.plot(cdf)
        plt.show(block=False)
        '''

        x_distr = margins(gradient_image.view(100, 50)[:, y].numpy())[0]

        points.append(grid_points_clone.view(100, 50, 3)[x, y].cpu().numpy())

    points = np.array(points)
    points = torch.from_numpy(points)
    return torch.tensor(points, dtype=torch.float32, device='cpu')  # grid_points_clone
