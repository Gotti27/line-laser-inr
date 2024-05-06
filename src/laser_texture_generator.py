import cv2 as cv
import numpy as np

texture = np.zeros((1000, 1000, 3), np.uint8)
cv.line(texture, (500, 0), (500, 1000), (0, 0, 255), 1)

cv.imshow("texture", texture)
cv.imwrite("data/laser.bmp", texture)
cv.waitKey(0)
cv.destroyAllWindows()
