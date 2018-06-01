'''
python command.py --n-img=21 --pretrained-folder=/Users/vincentwarmerdam/Desktop/pretrained
floyd run --cpu2 --env tensorflow-1.8 --data cantdutchthis/datasets/dataz/1:/datasets 'python command.py --n-img=21 --pretrained-folder=/output'
'''


import fire
import keras
import tensorflow as tf
from utils import logger, make_optimiser, get_pretrained_weights, final_layers_model
from settings import BASEPATH, PRETRAINED_PATH

def main(dataset='catdog-small', model="mobilenet", generator="random", n_img=100,
         epochs=20, batch_size=16, hidden_layer_size=3, optimiser="adam",
         learning_rate=0.0001, loss='binary_crossentropy',
         image_folder=None, pretrained_folder=None, output_folder=None):
    if not image_folder:
        image_folder = BASEPATH
    if not pretrained_folder:
        pretrained_folder = PRETRAINED_PATH
    logger.debug(f"keras {keras.__version__} running with tensorflow version: {tf.__version__}")
    x_train, y_train, x_valid, y_valid = get_pretrained_weights(dataset=dataset,
                                                                model=model,
                                                                generator=generator,
                                                                n_img=n_img,
                                                                image_folder=image_folder,
                                                                pretrained_folder=pretrained_folder)
    mod = final_layers_model(input_shape=x_train.shape, hidden_layer=hidden_layer_size)
    mod.summary()

    opt = make_optimiser(name=optimiser, learning_rate=learning_rate)
    mod.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    mod.fit(x_train, y_train,
            epochs=epochs, batch_size=batch_size,
            verbose=1, validation_data=(x_valid, y_valid))

def grid():
    pass

if __name__ == "__main__":
    fire.Fire({
        "main": main,
        "grid": grid
    })