import functools
from random import *

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications import resnet50, vgg16
from keras.preprocessing.image import ImageDataGenerator
import os
import datatools
import plotting


def ternaus_model_building(img_shape):
    inputs = layers.Input(shape=img_shape)
    model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    for key in layer_dict:
        print(key)
        print(layer_dict[key])
        print(layer_dict[key].output_shape)
        layer_dict[key].trainble = False

    num_filters = 32

    def decoder_block(input_tensor, concat_tensor, num_filters_a, num_filters_b, up_scale=2):
        decoder = layers.Conv2DTranspose(num_filters_a, (3, 3),
                                         strides=(up_scale, up_scale),
                                         padding='same')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Conv2D(num_filters_b, (3, 3), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        return decoder

    actual_inputs = layers.Conv2D(512, (3, 3), padding='same')(layer_dict['block5_pool'].output)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block5_conv3'].output, 256, 512)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block4_conv3'].output, 256, 512)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block3_conv3'].output, 128, 256)
    actual_inputs = decoder_block(actual_inputs, layer_dict['block2_conv2'].output, 64, 128)
    final = decoder_block(actual_inputs, layer_dict['block1_conv2'].output, 32, 1)

    model = models.Model(inputs=[inputs], outputs=[final])
    return model


def resnet_model_building(img_shape):
    inputs = layers.Input(shape=img_shape)
    model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    for key in layer_dict:
        print(key)
        print(layer_dict[key])
        print(layer_dict[key].output_shape)
        layer_dict[key].trainble = False

    tf.keras.utils.plot_model(
        model,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'
    )

    num_filters = 32

    def decoder_block(input_tensor, concat_tensor, num_filters, up_scale=2, cropping=None):
        decoder = layers.Conv2DTranspose(num_filters, (up_scale, up_scale),
                                         strides=(up_scale, up_scale),
                                         padding='same')(input_tensor)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        decoder = layers.Conv2D(num_filters * 2, (3, 3), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        if cropping is not None:
            decoder = layers.Cropping2D(((cropping, 0), (cropping, 0)))(decoder)
        decoder = layers.concatenate([decoder, concat_tensor], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation("relu")(decoder)
        return decoder

    actual_inputs = layers.MaxPooling2D((2, 2), strides=(2, 2))(layer_dict['add_15'].output)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn5c_branch2c'].output, 512)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn4f_branch2c'].output, 256)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn3d_branch2c'].output, 64)
    actual_inputs = decoder_block(actual_inputs, layer_dict['bn2c_branch2c'].output, 32)

    actual_inputs = layers.Conv2DTranspose(num_filters * 2, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters * 2 * 2, (3, 3), padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters * 2, (3, 3), padding='same')(actual_inputs)

    actual_inputs = layers.Conv2DTranspose(num_filters, (2, 2),
                                           strides=(2, 2),
                                           padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters * 2, (3, 3), padding='same')(actual_inputs)
    actual_inputs = layers.BatchNormalization()(actual_inputs)
    actual_inputs = layers.Activation("relu")(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters, (3, 3), padding='same')(actual_inputs)
    actual_inputs = layers.Conv2D(num_filters, (3, 3), padding='same')(actual_inputs)

    final = layers.Conv2D(1, (3, 3), padding='same', activation='softmax')(actual_inputs)

    model = models.Model(inputs=[inputs], outputs=[final])
    tf.keras.utils.plot_model(
        model,
        to_file='final_model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'
    )
    return model


def convolutional_model_building(img_shape,
                                 convolution_size,  # TODO Can be parametrized for each layer
                                 activation_layer,
                                 filters_nb_list,
                                 filters_scaling,
                                 filters_nb_center):
    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            input_tensor)
        print(encoder)
        encoder = layers.BatchNormalization()(encoder)
        print(encoder)
        encoder = layers.Activation(activation_layer)(encoder)
        print(encoder)
        encoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            encoder)
        print(encoder)
        encoder = layers.BatchNormalization()(encoder)
        print(encoder)
        encoder = layers.Activation(activation_layer)(encoder)
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
        decoder = layers.Activation(activation_layer)(decoder)
        decoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation(activation_layer)(decoder)
        decoder = layers.Conv2D(num_filters, (convolution_size, convolution_size), padding='same')(
            decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation(activation_layer)(decoder)
        return decoder

    inputs = layers.Input(shape=img_shape)

    encoders = [None] * len(filters_nb_list)

    actual_inputs = inputs
    for i in range(0, len(filters_nb_list)):
        a, b = encoder_block(actual_inputs, filters_nb_list[i], filters_scaling[i])
        encoders[i] = b
        actual_inputs = a

    center = conv_block(actual_inputs, filters_nb_center)

    actual_inputs = center
    for i in reversed(range(0, len(filters_nb_list))):
        print(encoders)
        actual_inputs = decoder_block(actual_inputs,
                                      encoders[i],
                                      filters_nb_list[i],
                                      filters_scaling[i])

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(actual_inputs)

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
    print("Y_TRUE: ", y_true)
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
    f1_score = - 2 * (precision * recall) / (precision + recall)
    return f1_score


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
                                      np.ceil(len(imgs_train) / float(batch_size))),
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


def load_model(model_save_path):
    return models.load_model(model_save_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                              'dice_loss': dice_loss,
                                                              'f1_score': f1_score})


def load_model_and_create_submission_file(model_save_path, csv_path):
    model = load_model(model_save_path)
    imgs = datatools.load_test_images('data/test_set_images/')
    predict_imgs = [predict_test_img(img, model) for img in imgs]
    create_csv(predict_imgs, csv_path)


def create_csv(grnd_list, csv_path):
    with open(csv_path, 'w') as f:
        f.write('id,prediction\n')
        for i, grnd in enumerate(grnd_list):
            for j in range(0, 38):
                for k in range(0, 38):
                    img_number = i + 1
                    label = np.round(np.mean(grnd[k * 16:k * 16 + 16, j * 16:j * 16 + 16]))
                    res = "{:03d}_{}_{},{}\n".format(img_number, j * 16, k * 16, int(label))
                    f.write(res)


def show_img(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='Greys_r')
    plt.show()


def predict_test_img(img, model):
    a = img[:400, :400]
    print("Shape: ", a.shape)
    b = img[:400, 208:]
    print("Shape: ", b.shape)
    c = img[208:, 208:]
    print("Shape: ", c.shape)
    d = img[208:, :400]
    print("Shape: ", d.shape)
    pred = model.predict(np.array(([a, b, c, d])))
    print("pred shape: ", pred.shape)
    e = np.zeros((608, 608, 1))
    e[:400, :400] = pred[0]
    e[:400, 208:] = pred[1]
    e[208:, 208:] = pred[2]
    e[208:, :400] = pred[3]
    return e


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
