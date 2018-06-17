'''
python command.py --n-img=21 --pretrained-folder=/Users/vincentwarmerdam/Desktop/pretrained
floyd run --cpu2 --env tensorflow-1.8 --data cantdutchthis/datasets/dataz/1:/datasets 'python command.py --n-img=21 --pretrained-folder=/output'
'''


import fire
import keras
import datetime as dt
import itertools as it
from uuid import uuid4 as uuid
import tensorflow as tf
import pandas as pd
from utils import logger, make_optimiser, get_pretrained_weights, final_layers_model, make_pretrained_weights
from settings import BASEPATH, PRETRAINED_PATH


class Metrics(keras.callbacks.Callback):
    def __init__(self, settings):
        self.settings = settings
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        data = {**logs, **self.settings, **{"epoch": self.epoch, "timestamp": str(dt.datetime.now())}}
        data.pop('output_folder')
        data.pop('pretrained_folder')
        data.pop('image_folder')
        self.losses.append(data)


def main(dataset='catdog', model="mobilenet", generator="random", n_train_img=100,
         n_orig_img=100, epochs=20, batch_size=16, hidden_layer_size=3, optimiser="adam",
         learning_rate=0.0001, loss='binary_crossentropy', early_stopping=True,
         image_folder=None, pretrained_folder=None, output_folder=None, use_tqdm=True):
    if not image_folder:
        image_folder = BASEPATH
    if not pretrained_folder:
        pretrained_folder = PRETRAINED_PATH
    inputs = locals()
    inputs['runid'] = str(uuid())[:8]

    logger.debug(f"keras {keras.__version__} running with tensorflow version: {tf.__version__}")
    weights = get_pretrained_weights(dataset=dataset, model=model, generator=generator,
                                     n_img=n_train_img, n_orig_img=n_orig_img, pretrained_folder=pretrained_folder)
    x_train, y_train, x_valid, y_valid = weights

    mod = final_layers_model(input_shape=x_train.shape, hidden_layer=hidden_layer_size)
    mod.summary()

    metrics = Metrics(inputs)
    callbacks = [metrics]
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5)
    if early_stopping:
        callbacks.append(stopping)

    opt = make_optimiser(name=optimiser, learning_rate=learning_rate)
    mod.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    logger.debug(f"model has compiled")
    mod.fit(x_train, y_train,
            epochs=epochs, batch_size=batch_size,
            verbose=1, validation_data=(x_valid, y_valid),
            callbacks=callbacks)
    logger.debug(f"training has completed!")
    df = pd.DataFrame([m for m in metrics.losses if type(m) == type(dict())])
    print(df)
    df.to_csv(f"/output/{inputs['runid']}.csv", index=False)


def run_grid(dataset="catdog", generator="random", model="mobilenet",
             pretrained_folder="/tmp", output_folder="/output", epochs=100, use_tqdm=True):
    models = [model]
    n_img_orig = [10, 100, 500, 1000, 2000]
    n_img_pretrained = [1000, 5000]
    combinations = it.product(models, n_img_orig, n_img_pretrained)
    combinations = [(mod, orig, n_gen) for mod, orig, n_gen in combinations if n_gen >= orig]
    logger.debug(f"starting a large grid with {combinations} combinations")
    for mod, orig, n_gen in combinations:
        logger.debug(f"setting for this run => n_orig:{orig}, n_train:{n_gen}")
        main(dataset=dataset, generator=generator, model=mod, output_folder=output_folder, early_stopping=False,
             n_orig_img=orig, n_train_img=n_gen, pretrained_folder=pretrained_folder, epochs=epochs, use_tqdm=use_tqdm)
        logger.debug(f"done with run.")


if __name__ == "__main__":
    fire.Fire({
        "single-experiment": main,
        "run-grid": run_grid
    })