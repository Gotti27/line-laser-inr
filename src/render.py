import math
import os
import pickle
import time

import cv2 as cv
import mitsuba as mi
import numpy as np
import torch

from utils import rotate_x, project_point, rotate_z

print(mi.variants())

if torch.cuda.is_available():
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant('scalar_rgb')
# mi.set_variant('llvm_ad_rgb')

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Execution flags
testing = False  # Just fooling the static analyzer :)
do_all_renders = True
debug = not torch.cuda.is_available()
laser_degree_delta = 30
target = 'Dragon-small'
target_mesh = target + '.ply'

# Calculating Camera Intrinsic parameters

vertical_fov = 60
img_width = 1024
img_height = 1024

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

import pyvista as pv

foo = []
bar = []
bel = []
foo.append([0, 0, 0])

if testing:
    testing_angle = 45
    scene = mi.load_file("scenes/gear_left.xml", angle=testing_angle, target=target_mesh,
                         laser_angle_delta=laser_degree_delta, res=256)

    image = mi.render(scene, spp=256)
    print(image)
    mi.util.write_bitmap(f"my_first_render_{0}.exr", image)
    time.sleep(1)
    # test = np.array(image[:, :, 3])  # cv.imread("my_first_render_0.exr", cv.IMREAD_UNCHANGED)
    render = cv.imread("my_first_render_0.exr", cv.IMREAD_UNCHANGED)
    print(render.shape)
    '''
    render = render * 255
    render[render > 255] = 255
    render = np.uint8(render)

    for i in range(render.shape[0]):
        for j in range(render.shape[1]):
            print(render[i][j][3])
            if render[i][j][3] == 0:
                render[i][j][0] = 255
                render[i][j][1] = 255
                render[i][j][2] = 255

    cv.imwrite("render-test.png", render[:, :, 0:3])
    '''

    # depth_map = render[:, :, 3]
    # normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_16U)
    # cv.imshow("Depth Map", normalized_depth_map)

    # t = np.array([0, 0, 7.])
    t = np.array([0, 0, 1.2])
    R = np.eye(3, 3)
    # R @= np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    ##### latest
    R @= rotate_x(30)
    ####
    R @= rotate_x(-90)
    # R @= rotate_y(testing_angle)
    R @= rotate_z(testing_angle)
    # R @= rotate_x(20)
    R @= np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    camera_position = - np.matrix(R).T @ t
    print("pose: ", camera_position)
    print("Projection Matrix: ", K @ np.concatenate([R, np.matrix(t).T], axis=1))

    print(np.append(camera_position, [[1]], axis=1))
    laser_center = np.squeeze(np.asarray(camera_position)) @ rotate_z(-laser_degree_delta)
    laser_norm = np.array([0.5, 0, 0]) @ rotate_z(-testing_angle) @ rotate_z(-laser_degree_delta)
    print("laser center: ", laser_center)
    print("laser norm: ", laser_norm)

    bel.append(np.squeeze(np.asarray(camera_position)))
    foo.append(np.squeeze(np.asarray(laser_norm)).tolist())
    foo.append([0, 0, 0])
    foo.append(np.squeeze(np.asarray(np.array([0, 0, 0.5]))).tolist())
    foo.append([0, 0, 0])
    foo.append(np.squeeze(np.asarray(laser_center)))
    bar.append(np.squeeze(np.asarray(laser_center)))

    points = [project_point(p, R, t, K) for p in
              [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [1, 1, 1], [0, 0, 0],
               (laser_norm + np.array([0, 0, 0.])).tolist()]]

    origin = points[0]
    top_x = points[1]
    top_y = points[2]
    top_z = points[3]
    # testing_point = points[4]
    translated_origin = points[5]
    norm_test = points[6]

    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_x[0])), int(round(top_x[1]))],
            [0, 0, 255], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_y[0])), int(round(top_y[1]))],
            [0, 255, 0], 1)
    cv.line(render, [int(round(origin[0])), int(round(origin[1]))], [int(round(top_z[0])), int(round(top_z[1]))],
            [255, 0, 0], 1)
    # cv.line(render, [int(round(origin[0])), int(round(origin[1]))],
    #        [int(round(testing_point[0])), int(round(testing_point[1]))],
    #        [255, 0, 255], 1)

    cv.line(render, [int(round(translated_origin[0])), int(round(translated_origin[1]))],
            [int(round(norm_test[0])), int(round(norm_test[1]))],
            [255, 255, 0], 1)

    cv.drawMarker(render, project_point([-0.5, 0.5, 0.5], R, t, K), [0, 255, 0])
    cv.drawMarker(render, project_point([0.5, 0.5, 0.5], R, t, K), [0, 255, 0])
    cv.drawMarker(render, project_point([0.5, 0.5, -0.5], R, t, K), [0, 255, 0])
    cv.drawMarker(render, project_point([-0.5, 0.5, -0.5], R, t, K), [0, 255, 0])

    cv.imshow("Render", render[:, :, 0:3])
    cv.waitKey(0)
    cv.destroyAllWindows()

    print(np.array(foo))
    plotter = pv.Plotter()
    plotter.add_lines(
        np.array(foo),
        width=10,
        color='blue'
    )
    plotter.add_points(
        pv.PolyData(foo),
        point_size=10,
    )
    plotter.add_points(
        pv.PolyData(bel),
        point_size=10,
        color='green'
    )
    plotter.add_points(
        pv.PolyData(bar),
        point_size=10,
        color='red'
    )
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()

    exit(0)


def do_renders(side):
    for i in range(0, 360):
        rendered_image = mi.render(
            mi.load_file(f"scenes/gear_{side}.xml", angle=i, target=target_mesh, laser_angle_delta=laser_degree_delta,
                         res=1024), spp=256)
        # cv.imshow("rendering progress", np.array(rendered_image))
        # cv.waitKey(1)
        mi.util.write_bitmap(f"renders/{target}/data_{i}_{side}_render.exr", rendered_image)

        print(f"{round(i / 360 * 100)}%")


if do_all_renders:
    if not os.path.exists(f"renders/{target}"):
        os.makedirs(f'renders/{target}')
    do_renders('right')
    do_renders('left')
    time.sleep(1)

image_folder = f'renders/{target}'

# process right images
right_images = [img for img in os.listdir(image_folder) if img.endswith(".exr") and ('right' in img)]
frame = cv.imread(os.path.join(image_folder, right_images[0]))
height, width, layers = frame.shape

right_images.sort(key=lambda name: int(name.split('_')[1]))

for degree, image in enumerate(right_images):
    # degree += 135
    render = image  # , position = image

    render = cv.imread(os.path.join(image_folder, render), cv.IMREAD_UNCHANGED)
    # position = cv.imread(os.path.join(image_folder, position), cv.IMREAD_UNCHANGED)

    _, red_render = cv.threshold(render[:, :, 2] * 255, 100, 255, cv.THRESH_BINARY)

    # t = np.array([0, 0, 7.])
    t = np.array([0, 0, 1.2])
    R = np.eye(3, 3)
    # R @= np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R @= rotate_x(-90)
    R @= rotate_z(degree)
    # R @= rotate_x(20)
    R @= np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    camera_position = - np.matrix(R).T @ t
    print("pose: ", camera_position)
    print("Projection Matrix: ", K @ np.concatenate([R, np.matrix(t).T], axis=1))

    print(np.append(camera_position, [[1]], axis=1))
    laser_center = np.squeeze(np.asarray(camera_position)) @ rotate_z(-laser_degree_delta)
    laser_norm = np.array([0.5, 0, 0]) @ rotate_z(-degree) @ rotate_z(-laser_degree_delta)
    print("laser center: ", laser_center)
    print("laser norm: ", laser_norm)

    with open(f'renders/{target}/data_{degree}_right.pkl', 'wb') as data_output_file:
        pickle.dump(K, data_output_file)
        pickle.dump(R, data_output_file)
        pickle.dump(t, data_output_file)
        pickle.dump(laser_center, data_output_file)
        pickle.dump(laser_norm, data_output_file)

    if debug:
        points = [project_point(p, R, t, K) for p in
                  [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
                   (laser_norm + np.array([0, 0, 0.])).tolist()]]

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

        cv.putText(render, "x", [int(round(top_x[0])), int(round(top_x[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255],
                   1)
        cv.putText(render, "y", [int(round(top_y[0])), int(round(top_y[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0],
                   1)
        cv.putText(render, "z", [int(round(top_z[0])), int(round(top_z[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0],
                   1)

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
    # degree += 135
    render = image  # , position = image

    render = cv.imread(os.path.join(image_folder, render), cv.IMREAD_UNCHANGED)
    # position = cv.imread(os.path.join(image_folder, position), cv.IMREAD_UNCHANGED)

    _, red_render = cv.threshold(render[:, :, 2] * 255, 100, 255, cv.THRESH_BINARY)

    # t = np.array([0, 0, 7.])
    t = np.array([0, 0, 1.2])
    R = np.eye(3, 3)
    # R @= np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    ##### latest
    R @= rotate_x(30)
    ####

    R @= rotate_x(-90)
    R @= rotate_z(degree)
    # R @= rotate_x(20)
    R @= np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    camera_position = - np.matrix(R).T @ t
    print("pose: ", camera_position)
    print("Projection Matrix: ", K @ np.concatenate([R, np.matrix(t).T], axis=1))

    print(np.append(camera_position, [[1]], axis=1))
    laser_center = np.squeeze(np.asarray(camera_position)) @ rotate_z(laser_degree_delta)
    laser_norm = np.array([0.5, 0, 0]) @ rotate_z(-degree) @ rotate_z(laser_degree_delta)
    print("laser center: ", laser_center)
    print("laser norm: ", laser_norm)

    with open(f'renders/{target}/data_{degree}_left.pkl', 'wb') as data_output_file:
        pickle.dump(K, data_output_file)
        pickle.dump(R, data_output_file)
        pickle.dump(t, data_output_file)
        pickle.dump(laser_center, data_output_file)
        pickle.dump(laser_norm, data_output_file)

    if debug:
        points = [project_point(p, R, t, K) for p in
                  [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
                   (laser_norm + np.array([0, 0, 0.])).tolist()]]

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

        cv.putText(render, "x", [int(round(top_x[0])), int(round(top_x[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255],
                   1)
        cv.putText(render, "y", [int(round(top_y[0])), int(round(top_y[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0],
                   1)
        cv.putText(render, "z", [int(round(top_z[0])), int(round(top_z[1]))], cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0],
                   1)

        # normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # cv.imshow("render depth", normalized_depth_map)
        # cv.imshow("positions", position)

        cv.imshow("render", render)
        cv.waitKey(1)

    time.sleep(1 / 60)
    # video.write(render)
    # video_depth.write(rendered_depth_map)

cv.destroyAllWindows()
