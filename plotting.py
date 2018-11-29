import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


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
