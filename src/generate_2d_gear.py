import random

from utils import *

image = np.zeros((500, 500, 1), np.uint8)
# cv.circle(image, (500, 500), 300, (255, 0, 0), 1)

for a in range(360):
    r = gear(a)
    cv.drawMarker(image, np.array(convert_polar_to_cartesian(a, r, (250, 250))).round().astype(int),
                  (200, 0, 0), cv.MARKER_CROSS, 1, 1)

cv.drawMarker(image, (250, 250), (100, 0, 0), cv.MARKER_CROSS, 10, 1)
# cv.drawMarker(blank_image, np.array(convert_polar_to_cartesian(0, 300, (500, 500))).round().astype(int), (200, 0, 0),
#              cv.MARKER_CROSS, 30, 5)

for _ in range(10):
    simulate_laser_rays(
        [random.randint(0, 500), random.randint(0, 500)],
        random.uniform(0., 360.), -1 if random.randint(0, 10) % 2 else 1, image)

p = np.array(convert_polar_to_cartesian(45, gear(45), (250, 250))).round().astype(int)
q = np.array([250, 300]).round().astype(int)
a = np.array([50, 50]).round().astype(int)
b = np.array([250, 250]).round().astype(int)
c = np.array([150, 245]).round().astype(int)
cv.drawMarker(image, p, (200, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1)
cv.drawMarker(image, q, (150, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1)
cv.drawMarker(image, a, (150, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1)
cv.drawMarker(image, b, (150, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1)
cv.drawMarker(image, c, (150, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1)
print("p", oracle(p))
print("q", oracle(q))
print("a", oracle(a))
print("b", oracle(b))
print("c", oracle(c))

cv.imshow("gear", image)

cv.waitKey(0)
cv.destroyAllWindows()
