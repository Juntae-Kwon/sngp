import sklearn.datasets
import torch
import numpy as np


DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_N_GRID = 100

def make_training_data(sample_size = 500):

    # Create two moon Training dataset
    train_examples, train_labels = sklearn.datasets.make_moons(n_samples=2 * sample_size, noise=0.1)

    # Adjust data position slightly
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]

    return torch.tensor(train_examples, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32)

def make_testing_data(x_range = DEFAULT_X_RANGE, y_range = DEFAULT_Y_RANGE, n_grid = DEFAULT_N_GRID):

    # Create a mesh grid in 2D space
    # Testing data (mesh grid over data space)
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    xv, yv = np.meshgrid(x, y)

    return torch.stack([torch.tensor(xv.flatten(), dtype=torch.float32), torch.tensor(yv.flatten(), dtype=torch.float32)], axis=-1)

def make_ood_data(sample_size = 500, means = (2.5, -1.75), vars = (0.01, 0.01)):
    return torch.tensor(np.random.multivariate_normal(means, cov=np.diag(vars), size=sample_size), dtype=torch.float32)
