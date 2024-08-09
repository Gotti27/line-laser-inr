import numpy as np


def conditional_x_2d(y, distribution):
    x_prob = distribution[:, y]
    x_prob = x_prob / np.sum(x_prob)
    # x_prob[np.isnan(x_prob)] = 0
    if np.sum(x_prob) > 0:
        return np.random.choice(len(x_prob), p=x_prob)
    else:
        return np.random.choice(len(x_prob))


def conditional_y_2d(x, distribution):
    y_prob = distribution[x, :]
    y_prob = y_prob / np.sum(y_prob)
    # y_prob[np.isnan(y_prob)] = 0
    if np.sum(y_prob) > 0:
        return np.random.choice(len(y_prob), p=y_prob)
    else:
        return np.random.choice(len(y_prob))


def gibbs_sampling_2d(distribution, num_samples, initial_point, values):
    original_distr = distribution

    distribution[np.isnan(distribution)] = 0
    distribution = distribution.flatten()
    distribution /= np.sum(distribution)

    samples = np.random.choice(len(distribution), p=distribution, size=num_samples)
    samples = [np.unravel_index(s, original_distr.shape) for s in samples]

    values = values.detach().cpu().view(50, 100, 3).numpy()
    return np.array([values[s[0], s[1]] for s in samples])


def conditional_x_3d(y, z, distribution):
    x_prob = distribution[:, y, z]
    x_prob /= x_prob.sum()
    if np.sum(x_prob) > 0:
        return np.random.choice(len(x_prob), p=x_prob)
    else:
        return np.random.choice(len(x_prob))


def conditional_y_3d(x, z, distribution):
    y_prob = distribution[x, :, z]
    y_prob /= y_prob.sum()
    if np.sum(y_prob) > 0:
        return np.random.choice(len(y_prob), p=y_prob)
    else:
        return np.random.choice(len(y_prob))


def conditional_z_3d(x, y, distribution):
    z_prob = distribution[x, y, :]
    z_prob /= z_prob.sum()
    if np.sum(z_prob) > 0:
        return np.random.choice(len(z_prob), p=z_prob)
    else:
        return np.random.choice(len(z_prob))


def gibbs_sampling_3d(distribution, num_samples, initial_point, values):
    original_distr = distribution

    distribution[np.isnan(distribution)] = 0
    distribution = distribution.flatten()
    distribution /= np.sum(distribution)

    samples = np.random.choice(len(distribution), p=distribution, size=num_samples)
    samples = [np.unravel_index(s, original_distr.shape) for s in samples]

    values = values.detach().view(100, 50, 100, 3).numpy()
    return np.array([values[s[0], s[1], s[2]] for s in samples])
