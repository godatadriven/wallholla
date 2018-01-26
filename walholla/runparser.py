import yaml
import pandas as pd

import walholla.models as models

import pytest


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
                                   output_dim,
                                   df.nodes,
                                   final_activation=None,
                                   activations=df.activation,
                                   dropouts=df.dropout)

    return model


def build_model(config):

    model_builders = {
        'fully-connected': build_fully_connected
    }

    model_config = config['model']
    return model_builders[model_config['type']](model_config)



