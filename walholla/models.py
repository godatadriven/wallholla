from keras.layers import Input, Dense, Activation, Add, Dropout
from keras.models import Model, Sequential

import numpy as np

import collections


def normal_model(input_dim, output_dim, depth, width, activation, final_activation, dropout):
    """
    This is a normal feed forward network.
    """
    return custom_fc_model(input_dim, output_dim, depth*[width], final_activation, activation, dropout)


def resnet_model(input_dim, output_dim, depth, width, activation, final_activation, dropout):
    """
    This is a feed forward architecture that contains skip connections. 
    This is similar to a resnet. The hope is that this makes the gradients 
    more stable during learning.
    """
    inputs = Input(shape=(input_dim,))
    nn = Dense(units=width, activation=activation)(inputs)
    for i in range(depth // 2):
        nn_odd = Dense(units=width, activation=activation)(nn)
        nn_even = Add()([nn, nn_odd])
        nn = Dense(units=width, activation=activation)(nn_even)
        if dropout > 0:
            nn = Dropout(dropout)(nn)
    nn = Dense(units=output_dim, activation=final_activation)(nn)
    return Model(inputs, nn)


def custom_fc_model(input_dim,
                    output_dim,
                    layer_definitions,
                    final_activation='linear',
                    activations="relu",
                    dropouts=0.0):
    """Create a custom, fully-connected model

    Args:
        input_dim (int)
            Shape/dimensionality of the input layer

        output_dim (int)
            Shape/dimensionality of the output layer

        layer_definitions (collections.Sized)
            For each layer the number of neurons for that layer

            The number of neurons can either be specified directly by specifying an
            integer number. When the number for a specific layer are given in float
            numbers, then the number of neurons for that layer are given relative
            to the input dimension. If the fraction does not yield a natural number,
            the number of neurons for that layer are computed by rounding the
            fractional number up to nearest integer. It is also possible to have
            mixed float and integer types.

            Be careful with 1 (integer: only one neuron) and 1.0 (float:
            as many neurons as inputs)

        final_activation (str)
            Specify the activation of the final layer

        activations (collections.Sized[str])
            Specify the activations for all layers or for each layer individually

            If the argument is str the same activation function is applied to all
            layers. If the argument is an iterable with str or Activation
            items, each layer can be specified individually. The number of
            activations specified should then match the number of layers specified
            in layer_definitions.

        dropouts (collections.Sized[float] or float)
            Specify the dropout probabilities globally or for each layer

    Examples:
        # Create a model with 16-dim input, feed it to a layer with 16 neurons
        # followed by 8 neurons and 2 neurons and let each neuron in each layer
        # have a dropout probability of 0.1
        custom_model(16, 1, [1.0, 0.5, 2], dropout=0.1)

        # Create a classification model that expands to twice the size of input,
        # then narrows down to output size. Dropout is only applied on the
        # expanding layers
        custom_model(2, 1, [1024, 8.0, 8, 2], dropout=[0.1, 0, 0, 0],
                     final_activation="sigmoid")

    Returns: Model
    """

    model = Sequential()
    input_shape = (input_dim, )

    if type(activations) == str:
        activations = len(layer_definitions)*[activations]
    if np.isscalar(dropouts):
        dropouts = len(activations)*[dropouts]

    for i, layer_def, activation, dropout in zip(range(len(layer_definitions)),
                                                 layer_definitions,
                                                 activations,
                                                 dropouts):
        if type(layer_def) == float:
            n_neurons = int(np.ceil(input_dim * layer_def))
            if n_neurons == 0:
                raise ValueError("Number of neurons rounded down to zero in layer ")
        else:
            n_neurons = layer_def

        if input_shape is not None:
            model.add(Dense(units=n_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(units=n_neurons, activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(units=output_dim, activation=final_activation))
    return model


