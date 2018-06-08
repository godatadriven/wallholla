'''
python command.py --n-img=21 --pretrained-folder=/Users/vincentwarmerdam/Desktop/pretrained
floyd run --cpu2 --env tensorflow-1.8 --data cantdutchthis/datasets/dataz/1:/datasets 'python command.py --n-img=21 --pretrained-folder=/output'
'''


import fire
import keras
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
        data = {**logs, **self.settings, **{"epoch": self.epoch}}
        data.pop('output_folder')
        data.pop('pretrained_folder')
        data.pop('image_folder')
        self.losses.append(data)


def main(dataset='catdog-small', model="mobilenet", generator="random", n_img=100,
         epochs=20, batch_size=16, hidden_layer_size=3, optimiser="adam",
         learning_rate=0.0001, loss='binary_crossentropy', early_stopping=None,
         image_folder=None, pretrained_folder=None, output_folder=None):
    if not image_folder:
        image_folder = BASEPATH
    if not pretrained_folder:
        pretrained_folder = PRETRAINED_PATH
    if not early_stopping:
        early_stopping = epochs
    inputs = locals()
    inputs['runid'] = str(uuid())[:8]

    logger.debug(f"keras {keras.__version__} running with tensorflow version: {tf.__version__}")
    x_train, y_train, x_valid, y_valid = get_pretrained_weights(dataset=dataset,
                                                                model=model,
                                                                generator=generator,
                                                                n_img=n_img,
                                                                image_folder=image_folder,
                                                                pretrained_folder=pretrained_folder)

    mod = final_layers_model(input_shape=x_train.shape, hidden_layer=hidden_layer_size)
    mod.summary()
    stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5)
    metrics = Metrics(inputs)

    opt = make_optimiser(name=optimiser, learning_rate=learning_rate)
    mod.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    logger.debug(f"model has compiled")
    mod.fit(x_train, y_train,
            epochs=epochs, batch_size=batch_size,
            verbose=1, validation_data=(x_valid, y_valid),
            callbacks=[stopping, metrics])
    logger.debug(f"training has completed!")
    print(pd.DataFrame([m for m in metrics.losses if type(m) == type(dict())]))


def make_pretraines(dataset="catdog", generator="random", model="vgg16", pretrained_folder="/output"):
    n_img_orig = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 5000]
    n_img_pretrained = [1000, 5000, 10000]
    for orig, n_gen in it.product(n_img_orig, n_img_pretrained):
        print(orig, n_gen)
        make_pretrained_weights(dataset=dataset, generator=generator,
                                class_mode="binary", model=model, n_train_img=n_gen,
                                img_size=(224, 224), pretrained_folder=pretrained_folder)


if __name__ == "__main__":
    fire.Fire({
        "single-experiment": main,
        "grid-pretrain": make_pretraines
    })