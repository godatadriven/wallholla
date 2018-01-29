from uuid import uuid4

import yaml
import pandas as pd
import datetime as dt
import numpy as np

import walholla.models as models

from keras.optimizers import Adam, RMSprop, SGD
from keras.losses import mean_squared_error, mean_absolute_error
from walholla.losses import boosted_loss

import walholla.datamakers as datamakers

import logging


def load_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f)

    return config


def build_fully_connected(model_config):

    input_dim = int(model_config['input_dim'])
    output_dim = int(model_config['output_dim'])

    df = pd.DataFrame.from_dict(model_config['layers'])
    df.nodes = df.nodes.replace({'output': output_dim})

    model = models.custom_fc_model(input_dim,
                                   df.nodes,
                                   activations=df.activation,
                                   dropouts=df.dropout)

    return model


def build_model(config):

    model_builders = {
        'fully-connected': build_fully_connected
    }

    model_config = config['model']
    return model_builders[model_config['type']](model_config)


def build_optimiser(config):

    opt_choices = {
        "adam": Adam,
        "rmsprop": RMSprop,
        "sgd": SGD
    }

    optimiser_config = config['optimiser']

    optimiser = None
    for algorithm, alg_arguments in optimiser_config.items():
        assert optimiser is None, "You have to specify exactly one optimiser"
        if type(alg_arguments) == dict:
            optimiser = opt_choices[algorithm](**alg_arguments)
        else:
            logging.info("Did not detect any optimisation parameters")
            optimiser = opt_choices[algorithm]()

    return optimiser


def build_loss(config):

    loss_config = config['loss']

    loss_choices = {
        "boosted_loss": boosted_loss,
        "mse": mean_squared_error,
        "mae": mean_absolute_error
    }

    return loss_choices[loss_config['type']]


def build_and_compile_model(config):

    model = build_model(config)

    loss = config['loss']['type']
    optimiser = build_optimiser(config)
    model.compile(loss=loss, optimizer=optimiser)

    return model


def build_dataset(config):

    datamaker_choices = {
        "checkerboard": datamakers.random_checkerboard,
        "normal_mirror": datamakers.random_normal_mirror,
        "binary_mirror": datamakers.random_binary_mirror
    }

    # TODO: code is somewhat similar to interpreting of optimiser
    dataset = None
    for algorithm, arguments in config['data'].items():
        assert dataset is None, "You have to specify exactly one data generator"
        if type(arguments) == dict:
            dataset = datamaker_choices[algorithm](**arguments)
        else:
            logging.info("Did not detect any optimisation parameters")
            dataset = datamaker_choices[algorithm]()

    return dataset


class Run(object):
    def __init__(self, config):
        self.config = config

        self.x, self.y = build_dataset(config)
        self.model = build_and_compile_model(config)

        self.fitpars = {'validation_split': 0.25,
                        'verbose': 2}
        if 'fitpars' in config:
            self.fitpars.update(config['fitpars'])

        np.random.seed(seed=config['global']['seed'])

        self.start_time = None
        self.end_time = None
        self.hist = None

    def execute(self):
        self.start_time = dt.datetime.now()
        self.hist = self.model.fit(self.x, self.y, **self.fitpars)
        self.end_time = dt.datetime.now()

        return self.get_run_summary()

    def get_run_summary(self):

        summary = {
            "seconds_taken": (self.end_time - self.start_time).total_seconds(),
            "epoch": range(1, 1 + len(np.array(self.hist.history["loss"]))),
            "num_samples": self.x.shape[0],
            "input_dim": self.x.shape[1],
            "output_dim": self.x.shape[0],
            "optimiser": self.model.optimizer,
            "loss_train": np.array(self.hist.history["loss"]),
            "loss_valid": np.array(self.hist.history["val_loss"]),
            "id": str(uuid4())[:8],
        }

        summary.update(self.fitpars)

        # TODO:
        # We may need to change these for custom layouts
        # "architecture": architecture,
        # "depth": depth,
        # "width": width,
        # "dropout": dropout,
        # "activation": activation

        # Depending on optimiser we have different parameters
        # summary.update(self.config['optimiser'])

        return pd.DataFrame(summary)
