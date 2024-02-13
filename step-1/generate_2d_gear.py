import math

import cv2 as cv
import numpy as np


def gear(angle):
    return 300 + (30 * math.sin(20 * math.radians(angle)))


def oracle(point):
    radius, angle = convert_cartesian_to_polar((500, 500), point)
    diff = radius - gear(angle)
    if abs(diff) < 0.3:
        return 0
    return np.sign(diff)


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


image = np.zeros((1000, 1000, 1), np.uint8)
# cv.circle(image, (500, 500), 300, (255, 0, 0), 1)

for a in range(361):
    r = gear(a)
    cv.drawMarker(image, np.array(convert_polar_to_cartesian(a, r, (500, 500))).round().astype(int),
                  (200, 0, 0), cv.MARKER_CROSS, 1, 2)

cv.drawMarker(image, (500, 500), (100, 0, 0), cv.MARKER_CROSS, 100, 2)
# cv.drawMarker(blank_image, np.array(convert_polar_to_cartesian(0, 300, (500, 500))).round().astype(int), (200, 0, 0),
#              cv.MARKER_CROSS, 30, 5)

p = np.array(convert_polar_to_cartesian(45, gear(45), (500, 500))).round().astype(int)
cv.drawMarker(image, p, (200, 0, 0), cv.MARKER_TILTED_CROSS, 50, 2)
print(oracle(p))

cv.imshow("gear", image)

cv.waitKey(0)
cv.destroyAllWindows()
