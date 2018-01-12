
import random
import fire
import datetime as dt
import numpy as np
import pandas as pd
from uuid import uuid4

import keras
import keras.backend as K
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import mean_squared_error, mean_absolute_error
from models import normal_model, resnet_model

def custom_loss(y_true, y_pred):
    when_i_dont_care = K.zeros(y_true.shape)
    when_i_do_care = K.log(y_true) - K.log(y_pred)
    return K.sum(K.where(y_true != y_pred, when_i_dont_care, when_i_do_care))

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='min')
base_logger = keras.callbacks.BaseLogger()
lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lambda x: 1/(1+x))
plateau_reduces = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

def experiment(epochs=2000,
               rows=2000,
               seed=42,
               depth=random.randint(1, 12),
               columns=random.randint(2, 10),
               width=np.random.randint(2, 10),
               dropout=random.choice([0, random.random()/3]),
               architecture=random.choice(["normal"]),
               optimiser = random.choice(["sgd", "adam", "rsmprop"]),
               loss = random.choice(['mse', 'mae']),
               learning_rate = np.power(10, - np.random.uniform(2, 5)),
               activation = random.choice(['linear', 'relu', 'softplus'])):
    np.random.seed(seed)

    OPT_CHOICES = {
        "adam": Adam(lr=learning_rate),
        "rsmprop": RMSprop(lr=learning_rate),
        "sgd": SGD(lr=learning_rate)
    }

    LOSS_FUNCTIONS = {
        "custom_loss": custom_loss,
        "mse": mean_squared_error,
        "mae": mean_absolute_error
    }

    MODELS = {
        "normal": normal_model,
        "resnet": resnet_model
    }

    data_train = np.random.normal(0, 1, size=(rows, columns))
    data_test = np.random.normal(0, 1, size=(rows, columns))

    model = MODELS[architecture](data_train, depth, width, activation, dropout)
    model.compile(optimizer=OPT_CHOICES[optimiser], loss=LOSS_FUNCTIONS[loss])
    start_time = dt.datetime.now()
    hist = model.fit(data_train, data_train,
                     validation_data=(data_test, data_test),
                     epochs=epochs, verbose=2, callbacks=[early_stopping, base_logger, lr_scheduler])
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
    # df.to_gbq('deep_experiment.mirrorfast',
    #           project_id='ml-babies',
    #           if_exists='append',
    #           private_key="/credentials/credentials.json")


if __name__ == "__main__":
    fire.Fire(experiment)
