import functools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import datatools
import plotting
from losses import dice_loss, bce_dice_loss, f1_score
from models import ternaus_model_building, resnet_model_building, convolutional_model_building
from submission import load_model, create_csv, predict_test_img


def train_model(val_rate, batch_size, improve_functions_list, epochs, model_type, model_save_path,
                optimizer,
                nb_imgs=100,
                resize_img=400,
                tpu=False,
                **kwargs):
    if model_type == 'unet':
        preprocess_input = None
    elif model_type == 'ternaus':
        preprocess_input = vgg16.preprocess_input
    elif model_type == 'resnet':
        preprocess_input = resnet50.preprocess_input
    else:
        preprocess_input = None

    imgs, grnd = datatools.load_training_images("data/training/", "images/", "groundtruth/",
                                                nb_imgs,
                                                resize_img, preprocess_input)
    imgs_test, imgs_train, grnd_test, grnd_train = datatools.split(imgs, grnd, val_rate)
    imgs_train, grnd_train = datatools.get_baseline_dataset(imgs_train, grnd_train,
                                                            improve_functions_list)

    if tpu:
        tf.keras.backend.clear_session()

    if model_type == 'unet':
        model = convolutional_model_building(imgs[0].shape, **kwargs)
    elif model_type == 'ternaus':
        model = ternaus_model_building(imgs[0].shape)
    elif model_type == 'resnet':
        model = resnet_model_building(imgs[0].shape)
    else:
        model = ternaus_model_building(imgs[0].shape)

    if tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(
                    tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
            )
        )

    model.compile(optimizer=optimizer, loss=bce_dice_loss,
                  metrics=[dice_loss, f1_score, 'accuracy'])
    model.summary()
    print("Trainable: ", model.trainable)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                      verbose=2,
                                                      monitor="val_f1_score",
                                                      save_best_only=True)

    datagen = ImageDataGenerator()

    history = model.fit_generator(datagen.flow(imgs_train, grnd_train, batch_size=batch_size),
                                  steps_per_epoch=int(np.ceil(len(imgs_train) / float(batch_size))),
                                  epochs=epochs,
                                  validation_data=datagen.flow(imgs_test, grnd_test),
                                  validation_steps=int(
                                      np.ceil(len(imgs_test) / float(batch_size))),
                                  callbacks=[checkpointer])

    return history, model, imgs, grnd


def train_model_and_plot_results(**kwargs):
    epochs = kwargs["epochs"]

    history, model, imgs, grnd = train_model(**kwargs)

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

    # pictures visualisation

    prediction = model.predict(imgs, steps=1)
    plotting.show_grnd(prediction[0])

    plt.figure(figsize=(10, 10))
    plt.imshow(imgs[0], cmap='Greys_r')
    plt.show()


def load_model_and_create_submission_file(model_save_path, csv_path):
    model = load_model(model_save_path)
    imgs = datatools.load_test_images('data/test_set_images/')
    predict_imgs = [predict_test_img(img, model) for img in imgs]
    create_csv(predict_imgs, csv_path)


def usage_example():
    config = {
        "batch_size": 8,
        "epochs": 200,
        "val_rate": 0.2,
        "nb_imgs": 100,
        "resize_img": 400,
        "model_save_path": "./model_weights.hdf5",
        "optimizer": optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                     amsgrad=False),

        # List of function that you will apply on all your data to improve it.
        "improve_functions_list": [
            functools.partial(datatools.flip_and_rotate, True, 0),
            functools.partial(datatools.flip_and_rotate, False, 45),
            functools.partial(datatools.flip_and_rotate, True, 45)
        ],
        "model_type": "unet",  # Model type: unet, ternaus, resnet

        # UNET PARAMETERS ONLY
        "convolution_size": 3,
        "activation_layer": 'relu',
        "filters_nb_list": [32, 64, 128, 256, 512],
        "filters_scaling": [2, 2, 2, 2, 5],
        "filters_nb_center": 1024
    }

    train_model_and_plot_results(**config)
