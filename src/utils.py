import math

import cv2 as cv
import numpy as np
from scipy import spatial


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
