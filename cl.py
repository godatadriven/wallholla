import datetime as dt
import random
from uuid import uuid4

import fire
import numpy as np
import pandas as pd
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import RMSprop, Adam, SGD
from walholla.models import normal_model, resnet_model
from walholla.callbacks import lr_scheduler, increasing_lr_scheduler, base_logger, early_stopping, plateau_reduces
from walholla.losses import boosted_loss
import walholla.datamakers as datamakers


ACTIVATIONS = ['linear', 'relu', 'softplus']

DATA_SOURCES = {
    "random_checkerboard": datamakers.random_checkerboard,
    "random_normal_mirror": datamakers.random_normal_mirror,
    "random_binary_mirror": datamakers.random_binary_mirror
}


def experiment(epochs=2000,
               rows=2000,
               seed=42,
               depth=random.randint(1, 12),
               columns=random.randint(2, 10),
               width=np.random.randint(2, 10),
               dropout=random.choice([0, random.random()/3]),
               architecture=random.choice(["normal"]),
               optimiser=random.choice(["sgd", "adam", "rmsprop"]),
               loss=random.choice(['mse', 'mae']),
               learning_rate=np.power(10, - np.random.uniform(2, 5)),
               activation=random.choice(ACTIVATIONS),
               data_source=random.choice(list(DATA_SOURCES.keys())),
               final_activation=None,
               batch_size=32):

    np.random.seed(seed)

    OPT_CHOICES = {
        "adam": Adam(lr=learning_rate),
        "rmsprop": RMSprop(lr=learning_rate),
        "sgd": SGD(lr=learning_rate)
    }

    LOSS_FUNCTIONS = {
        "boosted_loss": boosted_loss,
        "mse": mean_squared_error,
        "mae": mean_absolute_error
    }

    MODELS = {
        "normal": normal_model,
        "resnet": resnet_model
    }

    data_generator = DATA_SOURCES[data_source]

    if final_activation is None:
        final_activation = activation

    x_train, x_test, y_train, y_test = data_generator(n=rows, k=columns)

    input_dim, output_dim = x_train.shape[1], y_train.shape[1]

    model = MODELS[architecture](input_dim, output_dim, depth, width, activation, final_activation, dropout)

    model.compile(optimizer=optimiser, loss=LOSS_FUNCTIONS[loss])
    start_time = dt.datetime.now()
    hist = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     batch_size=batch_size,
                     epochs=epochs, verbose=2, callbacks=[base_logger, early_stopping]) # TODO: removed LR scheduling

    df = pd.DataFrame({
        "architecture": architecture,
        "time_end": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seconds_taken": (dt.datetime.now() - start_time).total_seconds(),
        "depth": depth,
        "epoch": range(1, 1 + len(np.array(hist.history["loss"]))),
        "max_epochs": epochs,
        "rows": rows,
        "columns": columns,
        "width": width,
        "dropout": dropout,
        "optimiser": optimiser,
        "learning_rate": learning_rate,
        "loss_train": np.array(hist.history["loss"]),
        "loss_valid": np.array(hist.history["val_loss"]),
        "activation": activation,
        "id": str(uuid4())[:8],
        "seed": seed
    })

    # TODO: add timestamp or something unique to filename, during development I actually like to overwrite
    # TODO: eventually disable and replace by GBQ insertion
    df.to_csv("result.csv")

    # df.to_gbq('deep_experiment.mirrorfast',
    #           project_id='ml-babies',
    #           if_exists='append',
    #           private_key="/credentials/credentials.json")


if __name__ == "__main__":
    fire.Fire(experiment)
