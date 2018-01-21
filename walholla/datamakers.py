"""
All functions here have an output of x_train, y_train, x_valid, y_valid
"""

import numpy as np

from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split

from functools import wraps

TEST_SIZE = 0.25


def tts_decorator(test_size=TEST_SIZE):
    def func_wrapper(data_generator):
        @wraps(data_generator)
        def wrap(*args, **kwargs):
            x, y = data_generator(*args, **kwargs)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
            return x_train, x_test, y_train, y_test
        return wrap
    return func_wrapper


@tts_decorator(test_size=TEST_SIZE)
def random_checkerboard(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    x = np.random.uniform(low=-1, high=1, size=[n, 2])
    y = np.sign(np.sin(k * x[:, 0] * np.pi) * np.sin(k * x[:, 1] * np.pi)).reshape(-1, 1)
    return x, y


@tts_decorator(test_size=TEST_SIZE)
def random_normal_mirror(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    x_train = np.random.normal(0, 1, size=(n, k))
    return x_train, x_train


@tts_decorator(test_size=TEST_SIZE)
def random_binary_mirror(n=10000, k=10, seed=42):
    np.random.seed(seed=seed)
    x_train = np.random.normal(0, 1, size=(n, k))
    return x_train, -x_train


@tts_decorator(test_size=TEST_SIZE)
def random_sklearn_classification(n=10000, k=10, seed=42, **kwargs):
    raise NotImplementedError("vincent was lazy")
    # np.random.seed(seed=seed)
    # make_classification(n_samples=n, n_features=k, random_state=seed, **kwargs)
