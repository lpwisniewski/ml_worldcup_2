import functools

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import datatools
import plotting
from losses import dice_loss, bce_dice_loss, f1_score
from tensorflow.python.keras import losses
from models import ternaus_model_building, resnet_model_building, convolutional_model_building
from submission import load_model, create_csv, predict_test_img
import scipy.misc

np.random.seed(1)


def train_model(batch_size, epochs, model_type, model_save_path,
                optimizer,
                loss_name="bce_dice",
                resize_img=400,
                tpu=False,
                **kwargs):
    """
    Train the model and save it in model_save_path. Look at the end of this file for more information
    about the parameters.
    """
    if model_type == 'unet':
        preprocess_input = None
    elif model_type == 'ternaus':
        preprocess_input = vgg16.preprocess_input
    elif model_type == 'resnet':
        preprocess_input = resnet50.preprocess_input
    else:
        preprocess_input = None

    img = datatools.load_image("./data/training/train/images/satImage_001.png")

    if tpu:
        tf.keras.backend.clear_session()

    if model_type == 'unet':
        model = convolutional_model_building(img.shape, **kwargs)
    elif model_type == 'ternaus':
        model = ternaus_model_building(img.shape)
    elif model_type == 'resnet':
        model = resnet_model_building(img.shape)
    else:
        model = ternaus_model_building(img.shape)

    if tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(
                    tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
            )
        )

    if loss_name == "bce_dice":
        loss = bce_dice_loss
    elif loss_name == "bce":
        loss = losses.binary_crossentropy
    elif loss_name == "dice":
        loss = dice_loss

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[dice_loss, f1_score, 'accuracy'])
    model.summary()
    print("Trainable: ", model.trainable)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                      verbose=2,
                                                      monitor="val_f1_score",
                                                      save_best_only=True)

    datagen = datatools.custom_image_generator("./data/training/train/images/",
                                               "./data/training/train/groundtruth/", batch_size,
                                               preprocess=preprocess_input)

    data_gen_val = datatools.custom_image_generator("./data/training/validation/images/",
                                                    "./data/training/validation/groundtruth/",
                                                    random=False, batch_size=batch_size)

    files = os.listdir("./data/training/train/images/")
    files_test = os.listdir("./data/test_set_images/")
    history = model.fit_generator(datagen,
                                  steps_per_epoch=int(np.ceil(len(files) / float(batch_size))),
                                  epochs=epochs,
                                  validation_data=data_gen_val,
                                  validation_steps=int(
                                      np.ceil(len(files_test) / float(batch_size))),
                                  callbacks=[checkpointer])

    return history, model


def train_model_and_plot_results(**kwargs):
    """
    Train the model according the given configuration (see the end of the file for example.
    """
    epochs = kwargs["epochs"]

    history, model = train_model(**kwargs)

    dice = history.history['dice_loss']
    val_dice = history.history['val_dice_loss']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='Training Dice Loss')
    plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Dice Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


def load_model_and_create_submission_file(model_save_path, csv_path):
    """
    This function is used to create easily a submission CSV from a registered model.
    """
    model = load_model(model_save_path)
    imgs = datatools.load_test_images('data/test_set_images/')
    predict_imgs = [predict_test_img(img, model) for img in imgs]
    create_csv(predict_imgs, csv_path)


def usage_example():
    config = {
        "batch_size": 2,
        "epochs": 400,
        "resize_img": 400,
        "model_save_path": "./model_weights.hdf5",
        # "optimizer": optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
        "optimizer": "adam",
        "loss_name": "dice_loss",
        # List of function that you will apply on all your data to improve it.
        "model_type": "unet",  # Model type: unet, ternaus, resnet

        # UNET PARAMETERS ONLY
        "convolution_size": 5,
        "activation_layer": 'relu',
        "filters_nb_list": [32, 64, 128, 256, 512],
        "filters_scaling": [2, 2, 2, 2, 5],
        "filters_nb_center": 1024
    }

    train_model_and_plot_results(**config)
