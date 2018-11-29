import functools
from random import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

import datatools
import plotting


def convolutional_model_building(img_shape):
    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        print(encoder)
        encoder = layers.BatchNormalization()(encoder)
        print(encoder)
        encoder = layers.Activation('relu')(encoder)
        print(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        print(encoder)
        encoder = layers.BatchNormalization()(encoder)
        print(encoder)
        encoder = layers.Activation('relu')(encoder)
        print(encoder)
        return encoder

    def encoder_block(input_tensor, num_filters, down_scale=2):
        encoder = conv_block(input_tensor, num_filters)
        print(encoder)
        encoder_pool = layers.MaxPooling2D((down_scale, down_scale),
                                           strides=(down_scale, down_scale))(encoder)
        print(encoder_pool)

        return encoder_pool, encoder

    def decoder_block(input_tensor, concat_tensor, num_filters, up_scale=2):
        decoder = layers.Conv2DTranspose(num_filters, (up_scale, up_scale),
                                         strides=(up_scale, up_scale),
                                         padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    inputs = layers.Input(shape=img_shape)
    # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512, 5)
    # 8
    center = conv_block(encoder4_pool, 1024)
    # center
    decoder4 = decoder_block(center, encoder4, 512, 5)
    # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def train_model(imgs, grnd, val_rate, batch_size, improve_function_list):
    imgs_test, imgs_train, grnd_test, grnd_train = datatools.split(imgs, grnd, val_rate)
    model = convolutional_model_building(imgs[0].shape)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, f1_score, 'accuracy'])
    model.summary()

    train_dataset = datatools.get_baseline_dataset(imgs_train,
                                                   grnd_train,
                                                   list_improve_func=improve_function_list)
    val_dataset = tf.data.Dataset.from_tensor_slices((imgs_test, grnd_test))
    val_dataset = val_dataset.repeat().batch(batch_size)
    # checkpointer = ModelCheckpoint(filepath='weights2.hdf5', verbose=2, save_best_only=True)
    history = model.fit(train_dataset,
                        steps_per_epoch=int(np.ceil(len(imgs_train) / float(batch_size))),
                        epochs=epochs,
                        validation_data=val_dataset,
                        validation_steps=int(np.ceil(len(imgs_test) / float(batch_size))))

    return history, model


def train_model_and_plot_results(batch_size, epochs, val_rate, improve_functions_list, imgs, grnd):
    history, model = train_model(imgs, grnd, val_rate, batch_size, improve_functions_list)

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

    a = randint(0, len(imgs))
    img = imgs[a]
    grnd = grnd[a]
    img_tensor = tf.convert_to_tensor([img], dtype=tf.float32)
    prediction = model.predict(img_tensor, steps=1)
    plotting.visualization_prediction(img, grnd, prediction[0])


batch_size = 10
epochs = 200
imgs, grnd = datatools.load_training_images("data/training/", "images/", "groundtruth/", 100)
val_rate = 0.2
# List of function that you will apply on all your data to improve it.
improve_function_list = [functools.partial(datatools.flip_img, True),
                         functools.partial(datatools.flip_img, False),
                         functools.partial(datatools.rotate_img, True),
                         functools.partial(datatools.rotate_img, False)]

train_model_and_plot_results(batch_size, epochs, val_rate, improve_function_list, imgs, grnd)
