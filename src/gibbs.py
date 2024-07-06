import numpy as np


def conditional_x_2d(y, distribution):
    x_prob = distribution[:, y]
    x_prob = x_prob / np.sum(x_prob)
    return np.random.choice(len(x_prob), p=x_prob)


def conditional_y_2d(x, distribution):
    y_prob = distribution[x, :]
    y_prob = y_prob / np.sum(y_prob)
    return np.random.choice(len(y_prob), p=y_prob)


def gibbs_sampling_2d(distribution, num_samples, initial_point, values):
    samples = np.zeros((num_samples, 2), dtype=int)
    current_point = np.array(initial_point)

    for i in range(num_samples):
        x = conditional_x_2d(current_point[1], distribution)
        y = conditional_y_2d(x, distribution)

        samples[i, :] = [x, y]
        current_point = [x, y]

    return np.array([values.detach().cpu().view(50, 100, 3).numpy()[s[0], s[1]] for s in samples])


def conditional_x_3d(y, z, distribution):
    x_prob = distribution[:, y, z]
    x_prob /= x_prob.sum()
    return np.random.choice(len(x_prob), p=x_prob)


def conditional_y_3d(x, z, distribution):
    y_prob = distribution[x, :, z]
    y_prob /= y_prob.sum()
    return np.random.choice(len(y_prob), p=y_prob)


def conditional_z_3d(x, y, distribution):
    z_prob = distribution[x, y, :]
    z_prob /= z_prob.sum()
    return np.random.choice(len(z_prob), p=z_prob)


def gibbs_sampling_3d(distribution, num_samples, initial_point, values):
    samples = np.zeros((num_samples, 3), dtype=int)
    current_point = initial_point

    for i in range(num_samples):
        x = conditional_x_3d(current_point[1], current_point[2], distribution)
        y = conditional_y_3d(x, current_point[2], distribution)
        z = conditional_z_3d(x, y, distribution)

        samples[i, :] = [x, y, z]
        current_point = [x, y, z]

    return np.array([values.detach().view(100, 50, 100, 3).numpy()[s[0], s[1], s[2]] for s in samples])
