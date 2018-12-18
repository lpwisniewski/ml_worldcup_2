import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt


def show_grnd(gt_img):
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
    gt_img8 = img_float_to_uint8(gt_img)
    gt_img_3c[:, :] = gt_img8
    gt_img_3c[:, :] = gt_img8
    gt_img_3c[:, :] = gt_img8

    plt.figure(figsize=(10, 10))
    plt.imshow(gt_img_3c, cmap='Greys_r')
    plt.show()


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
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def visualize_results_overlay(img, prediction):
    sess = tf.Session()
    with sess.as_default():
        im = img.eval()
    pred = np.squeeze(prediction, axis=2)
    cimg = concatenate_images(im, pred)
    plt.figure(figsize=(10, 10))  # create a figure with the default size
    plt.imshow(cimg, cmap='Greys_r')
    new_img = make_img_overlay(im, pred)
    plt.imshow(new_img)
    return


def visualize_results_concat(img, gt_img, prediction):
    sess = tf.Session()
    with sess.as_default():
        im = img.eval()
        gt_img = gt_img.eval()
    print(gt_img.shape)
    pred = np.squeeze(prediction, axis=2)
    gt_img = np.squeeze(gt_img, axis=2)
    cimg = concatenate_images(im, gt_img)
    cimg2 = concatenate_images(cimg, pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(cimg2, cmap='Greys_r')


def img_float_to_uint8(img):
    print(img.shape)
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def visualization_prediction(img, grnd, prediction):
    visualize_results_overlay(img, prediction)
    visualize_results_concat(img, grnd, prediction)


# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(image_id):
    imgwidth = 608
    imgheight = 608
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open("./results.csv")
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + 16, imgwidth)
        ie = min(i + 16, imgheight)
        if prediction == 0:
            adata = np.zeros((16, 16))
        else:
            adata = np.ones((16, 16))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')

    return im


def show_img(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='Greys_r')
    plt.show()
