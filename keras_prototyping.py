import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import tensorflow.keras.backend as K

# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    if len(data.shape) == 2:
        data = data[:,:,np.newaxis]
    result = tf.constant(data)
    return result


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def load_training_images(root_dir, path_real_images, path_ground_truth, max_img):
    """
    This function allows you to load easily training data.
    :param root_dir: Dir in which all data can be found
    :param path_real_images: Subfolder of root_dir where real images can be found
    :param path_ground_truth: Subfolder of root_dir where groundthruth images can be found
    :param max_img: Maximum number of images you want to load.
    :return: Returns the list of the images and their groundtruth in two separated matrix
    """
    image_dir = root_dir + path_real_images
    files = os.listdir(image_dir)
    n = min(max_img, len(files))  # Load maximum 20 images
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + path_ground_truth
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    return imgs, gt_imgs


def show_image_and_groundtruth(img, ground_truth):
    """
    Prints image next to it's groundtruth easily
    :param img: A matrix that represents the image
    :param ground_truth: A matrix taht represents the groundtruth
    :return: Nothing
    """
    # TODO rewrite for tensorflow input
    # cimg = concatenate_images(img, ground_truth)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cimg, cmap='Greys_r')


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


def main():
    batch_size = 6
    epochs = 200
    imgs, grnd = load_training_images("data/training/", "images/", "groundtruth/", 100)
    val_rate = 0.2
    val_split_number = int(len(imgs) * val_rate)
    imgs_test, imgs_train = imgs[:val_split_number], imgs[val_split_number:]
    grnd_test, grnd_train = grnd[:val_split_number], grnd[val_split_number:]
    print(len(imgs_train), len(imgs_test))
    show_image_and_groundtruth(imgs[0], grnd[0])

    print(imgs[0].shape)
    model = convolutional_model_building(imgs[0].shape)
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, f1_score])
    model.summary()

    train_dataset = tf.data.Dataset.from_tensor_slices((imgs_train, grnd_train))
    train_dataset = train_dataset.repeat().batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((imgs_test, grnd_test))
    val_dataset = val_dataset.repeat().batch(batch_size)
    history = model.fit(train_dataset,
                        steps_per_epoch=int(np.ceil(len(imgs_train) / float(batch_size))),
                        epochs=epochs,
                        validation_data=val_dataset,
                        validation_steps=int(np.ceil(len(imgs_test) / float(batch_size))))

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


main()
plt.show()
