import cv2 as cv

from utils import *

image = np.zeros((100, 100, 1), np.uint8)
# cv.circle(image, (500, 500), 300, (255, 0, 0), 1)

for a in range(361):
    r = gear(a)
    cv.drawMarker(image, np.array(convert_polar_to_cartesian(a, r, (50, 50))).round().astype(int),
                  (200, 0, 0), cv.MARKER_CROSS, 1, 1)

cv.drawMarker(image, (50, 50), (100, 0, 0), cv.MARKER_CROSS, 10, 1)
# cv.drawMarker(blank_image, np.array(convert_polar_to_cartesian(0, 300, (500, 500))).round().astype(int), (200, 0, 0),
#              cv.MARKER_CROSS, 30, 5)

p = np.array(convert_polar_to_cartesian(45, gear(45), (50, 50))).round().astype(int)
cv.drawMarker(image, p, (200, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1)
print(oracle(p))

cv.imshow("gear", image)

cv.waitKey(0)
cv.destroyAllWindows()
