import cv2 as cv
import numpy as np


def marching_squares(image, step, sigma):
    original = image.copy()
    vertexes = np.zeros((500 // step + 1, 500 // step + 1, 1), np.float32)
    for i in range(0, 500, step):
        for j in range(0, 500, step):
            value = original[i, j]
            # cv.drawMarker(image, [j, i], [100, 100, 100], cv.MARKER_CROSS, 2, 1)
            # cv.putText(image, str('0' if value >= sigma else '1'), [j, i],
            #          cv.FONT_HERSHEY_SIMPLEX, 0.5, [100, 100, 100], 1)
            vertexes[j // step, i // step] = 0 if value >= sigma else 1

    for y in range(0, 500 // step - 1):
        for x in range(0, 500 // step - 1):
            top_left = int(vertexes[x, y][0])
            top_right = int(vertexes[x + 1, y][0])
            bottom_right = int(vertexes[x + 1, y + 1][0])
            bottom_left = int(vertexes[x, y + 1][0])
            value = int(f'{top_left}{top_right}{bottom_right}{bottom_left}', 2)
            # cv.putText(image, str(value), [round((x * step) + (step / 2)), round((y * step) + (step // 2))],
            #          cv.FONT_HERSHEY_SIMPLEX, 0.5, [100, 100, 100], 1)
            draw_square(image, value, x, y, step)


def draw_square(image, value, x, y, step):
    match value:
        case 1 | 14:
            cv.line(image, [(x * step), (y * step) + (step // 2)], [(x * step) + (step // 2), (y * step) + step],
                    [150, 150, 150], 2)
        case 2 | 13:
            cv.line(image, [(x * step) + (step // 2), (y * step) + step], [(x * step) + step, (y * step) + (step // 2)],
                    [150, 150, 150], 2)
        case 3 | 12:
            cv.line(image, [(x * step), (y * step) + step // 2], [(x * step) + step, (y * step) + step // 2],
                    [150, 150, 150], 2)
        case 4 | 11:
            cv.line(image, [(x * step) + step // 2, (y * step)], [(x * step) + step, (y * step) + step // 2],
                    [150, 150, 150], 2)
        case 6 | 9:
            cv.line(image, [(x * step) + step // 2, (y * step)], [(x * step) + step // 2, (y * step) + step],
                    [150, 150, 150], 2)
        case 7 | 8:
            cv.line(image, [(x * step), (y * step) + step // 2], [(x * step) + step // 2, (y * step)],
                    [150, 150, 150], 2)
        case _:
            pass
