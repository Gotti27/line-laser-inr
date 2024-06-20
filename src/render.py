import math
import os
import pickle
import time

import cv2 as cv
import mitsuba as mi
import numpy as np

from src.utils import rotate_y, rotate_x, project_point

print(mi.variants())
mi.set_variant('llvm_ad_rgb')

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Execution flags
testing = True  # Just fooling the static analyzer :)
do_all_renders = False
laser_degree_delta = 30

# Calculating Camera Intrinsic parameters

vertical_fov = 60
img_width = 256
img_height = 256

fov_radians = math.radians(vertical_fov)
fx = fy = img_width / (2 * math.tan(fov_radians / 2))

Ox = img_width / 2.0
Oy = img_height / 2.0

K = np.array([
    [fx, 0, Ox],
    [0, fy, Oy],
    [0, 0, 1]
])

print(K)

###

if testing:
    testing_angle = 45
    scene = mi.load_file("scenes/gear_right.xml", angle=testing_angle)

    image = mi.render(scene, spp=256)
    print(image)
    mi.util.write_bitmap(f"my_first_render_{0}.exr", image)
    time.sleep(1)
    # test = np.array(image[:, :, 3])  # cv.imread("my_first_render_0.exr", cv.IMREAD_UNCHANGED)
    render = cv.imread("my_first_render_0.exr", cv.IMREAD_UNCHANGED)
    print(render.shape)
    # depth_map = render[:, :, 3]
    # normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_16U)
    # cv.imshow("Depth Map", normalized_depth_map)

    # t = np.array([0, 0, 7.])
    t = np.array([0, 2, 7.])
    R = np.eye(3, 3)
    R @= rotate_x(20)
    R @= rotate_y(testing_angle)
    # R @= rotate_x(90)

    camera_position = - np.matrix(R).T @ t
    print("pose: ", camera_position)
    print("Projection Matrix: ", K @ np.concatenate([R, np.matrix(t).T], axis=1))

    print(np.append(camera_position, [[1]], axis=1))
    laser_center = np.squeeze(np.asarray(camera_position)) @ rotate_y(laser_degree_delta)
    laser_norm = np.array([1, 0, 0]) @ rotate_x(20) @ rotate_y(testing_angle) @ rotate_y(laser_degree_delta)
    print("laser center: ", laser_center)
    print("laser norm: ", laser_norm)

    points = [project_point(p, R, t, K) for p in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0],
                                                  (laser_norm + np.array([0, 0, 0.])).tolist()]]

    origin = points[0]
    top_x = points[1]
    top_y = points[2]
    top_z = points[3]
    testing_point = points[4]
    translated_origin = points[5]
    norm_test = points[6]

    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_x[0])), int(round(top_x[1]))],
            [0, 0, 255], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_y[0])), int(round(top_y[1]))],
            [0, 255, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_z[0])), int(round(top_z[1]))],
            [255, 0, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))],
            [int(round(testing_point[0])), int(round(testing_point[1]))],
            [255, 0, 255], 1)

    cv.line(render, [int(round(translated_origin[0])), int(round(translated_origin[1]))],
            [int(round(norm_test[0])), int(round(norm_test[1]))],
            [255, 255, 0], 1)

    cv.imshow("Render", render[:, :, 0:3])
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit(0)


def do_renders(side):
    for i in range(360):
        rendered_image = mi.render(mi.load_file(f"scenes/gear_{side}.xml", angle=i), spp=256)
        cv.imshow("rendering progress", np.array(rendered_image))
        cv.waitKey(1)
        mi.util.write_bitmap(f"renders/data_{i}_{side}_render.exr", rendered_image)

        print(f"{round(i / 360 * 100)}%")


if do_all_renders:
    if not os.path.exists("renders"):
        os.mkdir('renders')
    do_renders('right')
    do_renders('left')
    time.sleep(1)

image_folder = 'renders'

# process right images
right_images = [img for img in os.listdir(image_folder) if img.endswith(".exr") and ('right' in img)]
frame = cv.imread(os.path.join(image_folder, right_images[0]))
height, width, layers = frame.shape

right_images.sort(key=lambda name: int(name.split('_')[1]))

for degree, image in enumerate(right_images):
    render = image  # , position = image

    render = cv.imread(os.path.join(image_folder, render), cv.IMREAD_UNCHANGED)
    # position = cv.imread(os.path.join(image_folder, position), cv.IMREAD_UNCHANGED)

    _, red_render = cv.threshold(render[:, :, 2] * 255, 100, 255, cv.THRESH_BINARY)

    t = np.array([0, 2, 7.])  # 1
    R = np.eye(3, 3)
    R @= rotate_x(20)
    R @= rotate_y(degree)

    camera_position = - np.matrix(R).T @ t
    print("pose: ", camera_position)
    print("Projection Matrix: ", K @ np.concatenate([R, np.matrix(t).T], axis=1))

    print(np.append(camera_position, [[1]], axis=1))
    laser_center = np.squeeze(np.asarray(camera_position)) @ rotate_y(laser_degree_delta)
    laser_norm = np.array([1, 0, 0]) @ rotate_x(20) @ rotate_y(degree) @ rotate_y(laser_degree_delta)
    print("laser center: ", laser_center)
    print("laser norm: ", laser_norm)

    with open(f'renders/data_{degree}_right.pkl', 'wb') as data_output_file:
        pickle.dump(K, data_output_file)
        pickle.dump(R, data_output_file)
        pickle.dump(t, data_output_file)
        pickle.dump(laser_center, data_output_file)
        pickle.dump(laser_norm, data_output_file)

    points = [project_point(p, R, t, K) for p in
              [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], (laser_norm + np.array([0, 0, 0.])).tolist()]]

    origin = points[0]
    top_x = points[1]
    top_y = points[2]
    top_z = points[3]
    norm_test = points[5]

    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_x[0])), int(round(top_x[1]))],
            [0, 0, 255], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_y[0])), int(round(top_y[1]))],
            [0, 255, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_z[0])), int(round(top_z[1]))],
            [255, 0, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))],
            [int(round(norm_test[0])), int(round(norm_test[1]))],
            [255, 255, 0], 1)

    cv.putText(render, "x", [int(round(top_x[0])), int(round(top_x[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 1)
    cv.putText(render, "y", [int(round(top_y[0])), int(round(top_y[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)
    cv.putText(render, "z", [int(round(top_z[0])), int(round(top_z[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)

    # normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # cv.imshow("render depth", normalized_depth_map)
    # cv.imshow("positions", position)

    cv.imshow("render", render)
    cv.waitKey(1)
    time.sleep(1 / 60)
    # video.write(render)
    # video_depth.write(rendered_depth_map)

## process left images
left_images = [img for img in os.listdir(image_folder) if img.endswith(".exr") and ('left' in img)]
frame = cv.imread(os.path.join(image_folder, left_images[0]))
height, width, layers = frame.shape

left_images.sort(key=lambda name: int(name.split('_')[1]))

for degree, image in enumerate(left_images):
    render = image  # , position = image

    render = cv.imread(os.path.join(image_folder, render), cv.IMREAD_UNCHANGED)
    # position = cv.imread(os.path.join(image_folder, position), cv.IMREAD_UNCHANGED)

    _, red_render = cv.threshold(render[:, :, 2] * 255, 100, 255, cv.THRESH_BINARY)

    t = np.array([0, 0, 7.])  # 1
    R = np.eye(3, 3)
    R @= rotate_x(0)
    R @= rotate_y(degree)

    camera_position = - np.matrix(R).T @ t
    print("pose: ", camera_position)
    print("Projection Matrix: ", K @ np.concatenate([R, np.matrix(t).T], axis=1))

    print(np.append(camera_position, [[1]], axis=1))
    laser_center = np.squeeze(np.asarray(camera_position)) @ rotate_y(-laser_degree_delta)
    laser_norm = np.array([1, 0, 0]) @ rotate_x(0) @ rotate_y(degree) @ rotate_y(-laser_degree_delta)
    print("laser center: ", laser_center)
    print("laser norm: ", laser_norm)

    with open(f'renders/data_{degree}_left.pkl', 'wb') as data_output_file:
        pickle.dump(K, data_output_file)
        pickle.dump(R, data_output_file)
        pickle.dump(t, data_output_file)
        pickle.dump(laser_center, data_output_file)
        pickle.dump(laser_norm, data_output_file)

    points = [project_point(p, R, t, K) for p in
              [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], (laser_norm + np.array([0, 0, 0.])).tolist()]]

    origin = points[0]
    top_x = points[1]
    top_y = points[2]
    top_z = points[3]
    norm_test = points[5]

    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_x[0])), int(round(top_x[1]))],
            [0, 0, 255], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_y[0])), int(round(top_y[1]))],
            [0, 255, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_z[0])), int(round(top_z[1]))],
            [255, 0, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))],
            [int(round(norm_test[0])), int(round(norm_test[1]))],
            [255, 255, 0], 1)

    cv.putText(render, "x", [int(round(top_x[0])), int(round(top_x[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 1)
    cv.putText(render, "y", [int(round(top_y[0])), int(round(top_y[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)
    cv.putText(render, "z", [int(round(top_z[0])), int(round(top_z[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)

    # normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # cv.imshow("render depth", normalized_depth_map)
    # cv.imshow("positions", position)

    cv.imshow("render", render)
    cv.waitKey(1)
    time.sleep(1 / 60)
    # video.write(render)
    # video_depth.write(rendered_depth_map)

cv.destroyAllWindows()
