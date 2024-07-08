import os
import pickle

import cv2 as cv
from torch.utils.data import Dataset


class INRPointsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label


def load_renders(images, target: str):
    renders = {}
    image_folder = f'renders/{target}'

    for image in images:
        degree = image.split('_')[1]
        side = image.split('_')[2]
        with open(f'renders/{target}/data_{degree}_{side}.pkl', 'rb') as data_input_file:
            K = pickle.load(data_input_file)
            R = pickle.load(data_input_file)
            t = pickle.load(data_input_file)
            laser_center = pickle.load(data_input_file)
            laser_norm = pickle.load(data_input_file)

        render_depth = cv.imread(os.path.join(image_folder, image), cv.IMREAD_UNCHANGED)
        renders[image] = {'K': K, 'R': R, 't': t,
                          'render': render_depth,
                          'laser_center': laser_center, 'laser_norm': laser_norm
                          }
    return renders
