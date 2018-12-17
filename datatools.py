import os

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import transform
from tensorflow.image import ResizeMethod


def load_image(infilename):
    data = mpimg.imread(infilename)
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    return data


def load_test_images(path_to_validation):
    files = os.listdir(path_to_validation)
    n = len(files)
    print("Loading " + str(n) + " test images")
    imgs = [load_image(path_to_validation + files[i]) for i in range(n)]
    return np.array(imgs)


def load_training_images(root_dir, path_real_images, path_ground_truth, max_img, resize,
                         preprocess=None):
    """
    This function allows you to load easily training data.
    :param preprocess: Preprocess function to apply to all images. Useful for pretrained models
    :param resize: New size of the image (images have to be square)
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
    if preprocess is not None:
        imgs = [preprocess(load_image(image_dir + files[i])) for i in range(n)]
    else:
        imgs = [load_image(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + path_ground_truth
    print("Loading " + str(n) + " ground truth")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    imgs = np.array([resize_img(resize, img) for img in imgs])
    gt_imgs = np.array([resize_img(resize, gr) for gr in gt_imgs])

    return imgs, gt_imgs


def split(imgs, grnd, validation_rate):
    """
    Split data according validation_rate
    """
    val_split_number = int(len(imgs) * validation_rate)
    imgs_test, imgs_train = imgs[:val_split_number], imgs[val_split_number:]
    grnd_test, grnd_train = grnd[:val_split_number], grnd[val_split_number:]
    return imgs_test, imgs_train, grnd_test, grnd_train


def flip_img(horizontal_flip, img, gt_img):
    if horizontal_flip:
        img, gt_img = np.fliplr(img), np.fliplr(gt_img)
    return img, gt_img


def rotate_img(angle, img, gt_img):
    return transform.rotate(img, angle), transform.rotate(gt_img, angle)


def flip_and_rotate(horizontal_flip, angle, img, gt_img):
    img, gt = flip_img(horizontal_flip, img, gt_img)
    return rotate_img(angle, img, gt)


def get_baseline_dataset(imgs,
                         labels,
                         list_improve_func=[]):
    """
    Get a full dataset with data improved.
    :param imgs: images in tensor form
    :param labels: groundtruth in the tensor form
    :param list_improve_func: List of function you want to apply on your data to improve it
    :param threads: Number of thread you want to create datasets and apply improve functions
    :return: A repeated dataset, batched and improved with given function.
    """

    # Apply all improvement function to our dataset and store results
    imgs_list = []
    labels_list = []
    for i in range(0, len(list_improve_func)):
        for j in range(0, len(imgs)):
            img, label = list_improve_func[i](imgs[j], labels[j])
            imgs_list.append(img)
            labels_list.append(label)

    return np.array(imgs_list), np.array(labels_list)


def resize_img(to_size, img):
    return transform.resize(img, (to_size, to_size))
