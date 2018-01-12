from keras.layers import Input, Dense, Activation, Add, Dropout
from keras.models import Model


def normal_model(data_train, depth, width, activation, dropout):
    """
    This is a normal feed forward network.
    """
    inputs = Input(shape=(data_train.shape[1],))
    nn = Dense(units=width, activation=activation)(inputs)
    for i in range(depth):
        nn = Dense(units=width, activation=activation)(nn)
        if dropout > 0:
            nn = Dropout(dropout)(nn)
    nn = Dense(units=data_train.shape[1], activation=activation)(nn)
    return Model(inputs, nn)

def resnet_model(data_train, depth, width, activation, dropout):
    """
    This is a feed forward architecture that contains skip connections. 
    This is similar to a resnet. The hope is that this makes the gradients 
    more stable during learning.
    """
    inputs = Input(shape=(data_train.shape[1],))
    nn = Dense(units=data_train.shape[1], activation=activation)(inputs)
    for i in range(depth // 2):
        nn_odd = Dense(units=width, activation=activation)(nn)
        nn_even = Add()([nn, nn_odd])
        nn = Dense(units=width, activation=activation)(nn_even)
        if dropout > 0:
            nn = Dropout(dropout)(nn)
    nn = Dense(units=data_train.shape[1], activation=activation)(nn)
    return Model(inputs, nn)