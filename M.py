from keras import backend as K
K.tensorflow_backend._get_available_gpus()

from google.colab import drive
drive.mount('/content/drive')
%cd  "/content/drive/My Drive/Colab Notebooks"
%pwd
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint
from random import*
import functools


def load_image(infilename):
    data = mpimg.imread(infilename)
    if len(data.shape) == 2:
        data = data[:,:,np.newaxis]
    result = tf.constant(data)
    return result


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
    outputs = layers.Conv2D(1, (1, 1), activation='softmax')(decoder0)

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


batch_size = 10
epochs = 100
imgs, grnd = load_training_images("data/training/", "images/", "groundtruth/", 100)
val_rate = 0.2
val_split_number = int(len(imgs) * val_rate)
imgs_test, imgs_train = imgs[:val_split_number], imgs[val_split_number:]
grnd_test, grnd_train = grnd[:val_split_number], grnd[val_split_number:]
print(len(imgs_train), len(imgs_test))

print(imgs[0].shape)
model = convolutional_model_building(imgs[0].shape)
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss, f1_score, 'accuracy'])
model.summary()

train_dataset = tf.data.Dataset.from_tensor_slices((imgs_train, grnd_train))
train_dataset = train_dataset.repeat().batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((imgs_test, grnd_test))
val_dataset = val_dataset.repeat().batch(batch_size)
#checkpointer = ModelCheckpoint(filepath='weights2.hdf5', verbose=2, save_best_only=True)
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

#pictures visualisation
from PIL import Image

def img_float_to_uint8(img):
    print(img.shape)
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        print(len(img.shape))
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

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def VisualizeResultsOverlay(img, prediction):
    sess = tf.Session()
    with sess.as_default():
        im = img.eval()
    w = img.shape[0]
    h = img.shape[1]
    pred = np.squeeze(prediction[0], axis=2)
    cimg = concatenate_images(im, pred)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')
    new_img = make_img_overlay(im, pred)
    plt.imshow(new_img)
    return
  
def VisualizeResultsConcat(img,gt_img,prediction):
    sess = tf.Session()
    with sess.as_default():
        im = img.eval()
        gt_img = gt_img.eval()
    print(gt_img.shape)
    pred = np.squeeze(prediction, axis=2)
    gt_img = np.squeeze(gt_img, axis=2)
    cimg = concatenate_images(im, gt_img)
    cimg2 = concatenate_images(cimg,pred)
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(cimg2, cmap='Greys_r')
  
  

a = randint(0,len(imgs))
img = imgs[a]
grnds = grnd[a]
print(img.shape)
grnds = grnd[a]
im = tf.convert_to_tensor([img], dtype=tf.float32)

prediction = model.predict(im, steps = 1)
print(len(prediction[0, 0]))

VisualizeResultsOverlay(img,prediction)
VisualizeResultsConcat(img,grnds,prediction[0])


#data_augmentation should go up while mking training dataset
import functools

def _augment(img, gt_img, horizontal_flip= randint(0,1), rotate_plus= randint(0,1)):
  
    img, gt_img = flip_img(horizontal_flip, img, gt_img)
    img, gt_img = rotate_img(rotate_plus, img, gt_img)
    return img, gt_img


def flip_img(horizontal_flip, img, gt_img):
    if horizontal_flip:
        tr_img, grd_img = tf.image.flip_left_right(img), tf.image.flip_left_right(gt_img)
    else:
        tr_img, grd_img = tf.image.flip_up_down(img), tf.image.flip_up_down(gt_img)
    return tr_img, grd_img


def rotate_img(rotate_plus, img, gt_img):
    if rotate_plus:
        tr_img, grd_img = tf.image.rot90(img,k=1), tf.image.rot90(gt_img,k=1)
    else:
        tr_img, grd_img = tf.image.rot90(img,k=3), tf.image.rot90(gt_img,k=3)
    return tr_img, grd_img

def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=batch_size,
                         shuffle=True):           
    num_x = len(filenames)
  # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  #dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  #if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
  #  assert batch_size == 1, "Batching images must be of the same size"
  
  

      dataset_2 = dataset.map(preproc_fn, num_parallel_calls=threads)
      print(dataset_2)
  
      dataset = dataset.concatenate(dataset_2)
  
      if shuffle:
        dataset = dataset.shuffle(num_x*2)
        print(dataset)
  
  
  # It's necessary to repeat our data for all epochs 
       dataset = dataset.repeat().batch(batch_size)
       return dataset

tr_cfg = {}
tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)


train_ds = get_baseline_dataset(imgs_train,
                                grnd_train,
                                preproc_fn=tr_preprocessing_fn,
                                batch_size=batch_size)
print(train_ds)
