import json
import os
import random

import numpy as np

from src.dataset import load_renders

NUMBER_IMAGES = 0

data = {
    "fl_x": 886.81,
    "fl_y": 886.81,
    "cx": 512,
    "cy": 512,
    "w": 1024.0,
    "h": 1024.0,
    "aabb_scale": 2,
    "frames": []
}

target = 'Dragon-mid-no-laser'
image_folder = f'renders/{target}'
counter = 0
images = [img for img in os.listdir(image_folder) if img.endswith(".exr")]
images.sort(key=lambda name: int(name.split('_')[1]))
images = random.sample(images, NUMBER_IMAGES if 0 < NUMBER_IMAGES < len(images) else len(images))

renders = load_renders(images, target, True)

for image in renders:
    print(image)
    roto_translation = np.concatenate(
        [np.concatenate([renders[image]['R'], np.matrix(renders[image]['t']).T * 2], axis=1), np.matrix([0, 0, 0, 1])],
        axis=0)

    data['frames'].append(
        {
            "file_path": 'images/' + image.split('.')[0] + '.png',
            "transform_matrix": roto_translation.tolist()
        },
    )

    counter += 1

with open('data/nerf.json', 'w') as fp:
    json.dump(data, fp)
