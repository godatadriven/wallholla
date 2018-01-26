"""
All functions here have an output of x_train, y_train, x_valid, y_valid
"""

import numpy as np

from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split

from functools import wraps


def random_checkerboard(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    x = np.random.uniform(low=-1, high=1, size=[n, 2])
    y = np.sign(np.sin(k * x[:, 0] * np.pi) * np.sin(k * x[:, 1] * np.pi)).reshape(-1, 1)
    return x, y


def random_normal_mirror(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    x_train = np.random.normal(0, 1, size=(n, k))
    return x_train, x_train


def random_binary_mirror(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    x_train = np.random.normal(0, 1, size=(n, k))
    return x_train, -x_train


def random_sklearn_classification(n=10000, k=10, seed=42, **kwargs):
    raise NotImplementedError("vincent was lazy")
    # np.random.seed(seed=seed)
    # make_classification(n_samples=n, n_features=k, random_state=seed, **kwargs)
