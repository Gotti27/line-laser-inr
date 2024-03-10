import math

import numpy as np


def gear(angle):
    return 100 + (10 * math.sin(20 * math.radians(angle)))


def oracle(point):
    radius, angle = convert_cartesian_to_polar((250, 250), point)
    diff = radius - gear(angle)
    return diff  # np.sign(diff)


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
