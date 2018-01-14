"""
All functions here have an output of x_train, y_train, x_valid, y_valid
"""

import numpy as np

from sklearn.datasets import make_blobs, make_classification, make_regression


def random_checkerboard(n=10000, bins=10, seed=42):
    np.random.seed(seed=seed)
    x = np.random.uniform(low=-1, high=1, size=[n, 2])
    y = np.sign(np.sin(bins * x[:, 0] * np.pi) * np.sin(bins * x[:, 1] * np.pi))
    y += x[:, 0] + x[:, 1]
    return x, y, x, y

def random_normal_mirror(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    data_train = np.random.normal(0, 1, size=(n, k))
    data_test = np.random.normal(0, 1, size=(n, k))
    return data_train, data_train, data_test, data_test

def random_binary_mirror(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    data_train = np.random.randint(0, 2, size=(n, k))
    data_test = np.random.normal(0, 2, size=(n, k))
    return data_train, data_train, data_test, data_test

def random_sklearn_classification(n=10000, k=10, seed=42, **kwargs):
    raise NotImplementedError("vincent was lazy")
    # np.random.seed(seed=seed)
    # make_classification(n_samples=n, n_features=k, **kwargs)