from walholla.models import custom_fc_model

import numpy as np

import pytest


def test_custom_fc_model():
    """Test happy flow for creating a custom model"""
    model = custom_fc_model(16, [1.0, 0.5, 2, 1])

    weight_matrix_dims = [(16, 16), (16, 8), (8, 2), (2, 1)]
    bias_dims = [16, 8, 2, 1]

    for i_layer, expected_dim_weight, expected_dim_bias in zip(range(4), weight_matrix_dims, bias_dims):
        weights, biases = model.layers[i_layer].get_weights()
        assert np.array(weights).shape == expected_dim_weight
        assert np.array(biases).shape[0] == expected_dim_bias
