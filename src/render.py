import os
import time

import cv2 as cv
import drjit as dr
import mitsuba as mi

print(mi.variants())
mi.set_variant('llvm_ad_rgb')

scene = mi.load_file("scenes/gear.xml")

image = mi.render(scene, spp=256)

print(image)
mi.util.write_bitmap(f"my_first_render_{0}.exr", image)
time.sleep(1)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

testing = False  # Just fooling the static analyzer :)
if testing:
    # test = np.array(image[:, :, 3])  # cv.imread("my_first_render_0.exr", cv.IMREAD_UNCHANGED)
    render = cv.imread("my_first_render_0.exr", cv.IMREAD_UNCHANGED)
    print(render.shape)
    depth_map = render[:, :, 3]
    normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_16U)

    cv.imshow("Depth Map", normalized_depth_map)
    cv.imshow("Render", render)
    cv.waitKey(0)
    cv.destroyAllWindows()

    exit(0)

do_all_renders = False
if do_all_renders:
    for i in range(360):
        params = mi.traverse(scene)
        rot = mi.Transform4f.rotate(axis=mi.Point3f([0, 0, 1]), angle=1)
        v = dr.unravel(mi.Point3f, params['teapot.vertex_positions'])
        params.update({
            'teapot.vertex_positions': dr.ravel(rot @ v)
        })

        image = mi.render(scene, spp=256)

        mi.util.write_bitmap(f"renders/my_first_render_{i}.exr", image)
        print(f"{round(i / 360 * 100)}%")

    time.sleep(1)

image_folder = 'renders'
video_name = 'render.mp4'
video_depth_name = 'renderDepth.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
frame = cv.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*'MP4V'), 20.0, (width, height))
video_depth = cv.VideoWriter(video_depth_name, cv.VideoWriter_fourcc(*'MP4V'), 20.0, (width, height))
images.sort(key=lambda name: int(name.split('_')[3].split('.')[0]))

for image in images:
    frame = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)
    render = frame[:, :, 0:3]
    depth_map = frame[:, :, 3]
    normalized_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    cv.imshow("render", render)
    cv.imshow("render depth", normalized_depth_map)
    cv.waitKey(1)
    time.sleep(1 / 60)
    # video.write(render)
    # video_depth.write(rendered_depth_map)

cv.destroyAllWindows()
video.release()
video_depth.release()
