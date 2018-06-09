import sys
import os
import glob
import logging
from collections import Counter
import datetime as dt

import tqdm
import fire
import numpy as np
from shutil import copyfile, rmtree
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model, Sequential
from keras import applications
from keras import backend as kb

from settings import BASEPATH, PRETRAINED_PATH, TMP_FOLDER


stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(levelname)s - %(message)s',
    handlers=[stdout_handler]
)
logger = logging.getLogger('keras')


def ensure_exists(fullpath):
    dirpath, filename = os.path.split(fullpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return fullpath


def make_optimiser(name="adam", learning_rate=0.0001, **kwargs):
    """Helper to create an optimiser from a string."""
    if name not in ["adam", "rsmprop", "sgd"]:
        raise ValueError("name needs to be either adam, rsmprop or sgd")
    optimizers = {
        "adam": Adam(lr=learning_rate, **kwargs),
        "rsmprop": RMSprop(lr=learning_rate, **kwargs),
        "sgd": SGD(lr=learning_rate, **kwargs)
    }
    logger.debug(f"optimiser is of type {name} with lr {learning_rate}")
    return optimizers[name]


def get_pretrained_model(model, img_size):
    """Helper to retreive appropriate pretrained model"""
    possible_names = ["vgg16", "vgg19", "mobilenet", "xception"]
    if model not in possible_names:
        raise ValueError(f"model needs to be either {possible_names}")
    logger.debug(f"backend for model is {model}")
    width, height = img_size
    if kb.image_data_format() == 'channels_first':
        input_shape = (3, width, height)
    else:
        input_shape = (width, height, 3)
    logger.debug(f"backend suggests that we have the following input shape: {input_shape}")
    models = {
        "vgg16": applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet'),
        "vgg19": applications.VGG19(input_shape=input_shape, include_top=False, weights='imagenet'),
        "mobilenet": applications.MobileNet(input_shape=input_shape, include_top=False, weights='imagenet'),
        "xception": applications.Xception(input_shape=input_shape, include_top=False, weights='imagenet')
    }
    return models[model]


def get_image_generator(kind="random"):
    if kind not in ["random", "very-random", "not-random"]:
        raise ValueError("kind needs to be in `random`, `very-random`, `not-random`")
    logger.debug(f"making image generator {kind}")
    if kind == "not-random":
        return ImageDataGenerator(rescale=1. / 255)
    if kind == "random":
        return ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.1)
    if kind == "very-random":
        return ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)


def get_folders(dataset="catdog"):
    train_data_dir = os.path.join(BASEPATH, dataset, 'train')
    validation_data_dir = os.path.join(BASEPATH, dataset, 'validation')
    logger.debug(f"train_data_dir: {train_data_dir}")
    logger.debug(f"validation_data_dir: {validation_data_dir}")
    return train_data_dir, validation_data_dir


def make_pretrained_filenames(dataset, model, generator, n_img, img_size, n_orig_img, npy=True):
    size_str = "x".join([str(_) for _ in img_size])
    names = [
        f"{dataset}/{model}/{generator}/{size_str}-{n_orig_img}-{n_img}-train-data",
        f"{dataset}/{model}/{generator}/{size_str}-{n_orig_img}-{n_img}-train-label",
        f"{dataset}/{model}/{size_str}-valid-data",
        f"{dataset}/{model}/{size_str}-valid-label",
    ]
    if npy:
        return [_ + '.npy' for _ in names]
    return names


def copy_files_tmp(n_orig_img, train_folder):
    tmp_folder = TMP_FOLDER
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    logger.debug(f"copying {n_orig_img} imgs to {tmp_folder}")
    train_class_dirs = glob.glob(f"{train_folder}/*/")
    for imgdir in train_class_dirs:
        class_img_paths = glob.glob(imgdir + '*')
        logger.debug(f"found class dirs: {imgdir} with {len(class_img_paths)} imgs")
        for path_from in class_img_paths[:n_orig_img//2]:
            path, filename = os.path.split(path_from)
            path_to = os.path.join(tmp_folder, os.path.basename(path), filename)
            dir_to = os.path.join(tmp_folder, os.path.basename(path))
            if not os.path.exists(dir_to):
                os.mkdir(dir_to)
            copyfile(path_from, path_to)
        logger.debug(f"{dir_to} now contains {len(glob.glob(dir_to + '/*'))} files")
    return tmp_folder


def make_validation_pretrained(dataset, class_mode, model, pretrained_folder, filenames, img_size):
    x_train_fname, y_train_fname, x_valid_fname, y_valid_fname = filenames
    logger.debug("will make pretrained validation set")
    datagen = get_image_generator(kind="not-random")
    train_folder, valid_folder = get_folders(dataset=dataset)

    valid_generator = datagen.flow_from_directory(
        valid_folder,
        target_size=img_size,
        batch_size=1,
        class_mode=class_mode)
    n_imgs = len(glob.glob(valid_folder + '/*/*'))
    x_shape = [n_imgs ] + list(valid_generator.image_shape)
    y_shape = (n_imgs ,)
    x_valid = np.ones(shape=x_shape)
    y_valid = np.ones(shape=y_shape)
    logger.debug(f"validation data to have shape {x_valid.shape}")
    logger.debug(f"validation labels to have shape {y_valid.shape}")
    for i in tqdm.tqdm(range(len(glob.glob(valid_folder + '/*/*')))):
        img_data_valid, label_data_valid = next(valid_generator)
        x_valid[i, :] = img_data_valid.squeeze()
        y_valid[i] = label_data_valid.squeeze()
    logger.debug(f"valid labels have following counts: {Counter(np.sort(y_valid))}")

    base_model = get_pretrained_model(model=model, img_size=img_size)
    logger.debug(f"about to apply {model} to validation data")
    tick = dt.datetime.now()
    pretrained_valid = base_model.predict(x_valid, verbose=1)
    logger.debug(f"predicting took {(dt.datetime.now() - tick).seconds}s or {(dt.datetime.now() - tick)} time")
    data_fp_x_valid = ensure_exists(os.path.join(pretrained_folder, x_valid_fname))
    data_fp_y_valid = ensure_exists(os.path.join(pretrained_folder, y_valid_fname))
    np.save(data_fp_x_valid, pretrained_valid)
    logger.debug(f"data has been written over at {data_fp_x_valid}")
    np.save(data_fp_y_valid, y_valid)
    logger.debug(f"data has been written over at {data_fp_y_valid}")


def make_training_pretrained(datagen, dataset, class_mode, model, pretrained_folder, filenames, n_orig_img, n_train_img, img_size):
    x_train_fname, y_train_fname, x_valid_fname, y_valid_fname = filenames
    train_folder, valid_folder = get_folders(dataset=dataset)
    if not n_orig_img:
        n_orig_img = len(glob.glob(train_folder+'*/*'))
        logger.debug(f"`n_orig_img` not set, will take all {n_orig_img} imgs in train folder")

    logger.debug(f"will take set of {n_orig_img} imgs for training")
    train_folder = copy_files_tmp(n_orig_img=n_orig_img, train_folder=train_folder)

    train_generator = datagen.flow_from_directory(
        train_folder,
        target_size=img_size,
        batch_size=1,
        class_mode=class_mode)
    x_shape = [n_train_img] + list(train_generator.image_shape)
    y_shape = (n_train_img,)
    logger.debug("about to generate datasets for training-set of pretrained model")
    x_train = np.ones(shape=x_shape)
    y_train = np.ones(shape=y_shape)

    logger.debug(f"train data to have shape {x_train.shape}")
    logger.debug(f"train labels to have shape {y_train.shape}")
    logger.debug(f"about to generate pretrained dataset for training")
    for i in tqdm.tqdm(range(n_train_img)):
        img_data_train, label_data_train = next(train_generator)
        x_train[i, :] = img_data_train.squeeze()
        y_train[i] = label_data_train.squeeze()
    logger.debug(f"train labels have following counts: {Counter(np.sort(y_train))}")

    base_model = get_pretrained_model(model=model, img_size=img_size)
    logger.debug(f"about to apply {model} to train data")
    tick = dt.datetime.now()
    pretrained_train = base_model.predict(x_train, verbose=1)

    logger.debug(f"predicting took {(dt.datetime.now() - tick).seconds}s or {(dt.datetime.now() - tick)} time")
    data_fp_x_train = ensure_exists(os.path.join(pretrained_folder, x_train_fname))
    data_fp_y_train = ensure_exists(os.path.join(pretrained_folder, y_train_fname))
    np.save(data_fp_x_train, pretrained_train)
    logger.debug(f"data has been written over at {data_fp_x_train}")
    np.save(data_fp_y_train, y_train)
    logger.debug(f"data has been written over at {data_fp_y_train}")

def make_pretrained_weights(dataset="catdog", generator="random", class_mode="binary",
                            model="vgg16", n_train_img=100, img_size=(224, 224),
                            pretrained_folder=PRETRAINED_PATH, n_orig_img=None):
    filenames = make_pretrained_filenames(dataset, generator, model, n_train_img, img_size, n_orig_img=n_orig_img, npy=False)
    x_train_fname, y_train_fname, x_valid_fname, y_valid_fname = filenames
    logger.debug(f"filename x_train_fname = {x_train_fname}")
    logger.debug(f"filename y_train_fname = {y_train_fname}")
    logger.debug(f"filename x_valid_fname = {x_valid_fname}")
    logger.debug(f"filename y_valid_fname = {y_valid_fname}")
    logger.debug("about to make pretrained weights")

    datagen = get_image_generator(kind=generator)
    make_training_pretrained(datagen=datagen,
                             dataset=dataset,
                             class_mode=class_mode,
                             model=model,
                             pretrained_folder=pretrained_folder,
                             filenames=filenames,
                             n_train_img=n_train_img,
                             img_size=img_size,
                             n_orig_img=n_orig_img)

    make_validation_pretrained(dataset=dataset,
                               class_mode=class_mode,
                               model=model,
                               pretrained_folder=pretrained_folder,
                               filenames=filenames,
                               img_size=img_size)




    if n_orig_img:
        rmtree(TMP_FOLDER)
        logger.debug("cleaned up tmp folder after writing pretrained features")


def get_pretrained_weights(dataset="catdog", generator="random", class_mode="binary",
                           model="mobilenet", n_img=10, img_size=(224, 224),
                           pretrained_folder=PRETRAINED_PATH):
    filenames = make_pretrained_filenames(dataset, generator, model, n_img, img_size, npy=True)
    for name in filenames:
        if not os.path.exists(os.path.join(pretrained_folder, name)):
            logger.debug(f"{os.path.join(pretrained_folder, name)} does not exist! creating .npz files.")
            make_pretrained_weights(dataset=dataset, generator=generator, class_mode=class_mode,
                                    model=model, n_train_img=n_img, img_size=img_size,
                                    pretrained_folder=pretrained_folder)

    x_train_fpath, y_train_fpath, x_valid_fpath, y_valid_fpath = [os.path.join(pretrained_folder, _) for _ in filenames]
    x_train = np.load(x_train_fpath)
    y_train = np.load(y_train_fpath)
    x_valid = np.load(x_valid_fpath)
    y_valid = np.load(y_valid_fpath)
    logger.debug("pretrained weight files have been found and loaded")
    for i in [os.path.join(PRETRAINED_PATH, _) for _ in filenames]:
        logger.debug(f"loaded file {i}")
    logger.debug(f"train data has shape: {x_train.shape}")
    logger.debug(f"valid data has shape: {x_valid.shape}")
    logger.debug(f"train labels have following counts: {Counter(np.sort(y_train))}")
    logger.debug(f"valid labels have following counts: {Counter(np.sort(y_valid))}")
    return x_train, y_train, x_valid, y_valid


def final_layers_model(input_shape, hidden_layer=10, dropout=0.5):
    logger.debug(f"creating model for image shape {input_shape}")
    img_input = Input(shape=input_shape[1:])
    x = Flatten()(img_input)
    x = Dense(hidden_layer, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation="sigmoid", name="superfinallayer")(x)
    model = Model(img_input, x)
    return model


if __name__ == "__main__":
    fire.Fire(make_pretrained_weights)
