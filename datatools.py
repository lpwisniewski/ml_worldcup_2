import os

import matplotlib.image as mpimg
import numpy as np
import plotting
import functools
import matplotlib
from skimage import transform
from tensorflow.image import ResizeMethod
from PIL import Image


def load_image(infilename):
    data = mpimg.imread(infilename)
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    return data


def load_test_images(path_to_validation):
    """
    Load images to create CSV
    """
    files = os.listdir(path_to_validation)
    files = list(map(lambda t: t[1],
                     sorted(list(zip([int(f.split("_")[1].split(".")[0]) for f in files], files)),
                            key=lambda tup: tup[0])))
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


def concatenate_image(img, gt):
    """Mirrors and concatenates images"""
    img_fv = np.flipud(img)
    gt_fv = np.flipud(gt)
    img_fh, gt_fh = np.fliplr(img), np.fliplr(gt)
    h = img.shape[0]
    w = img.shape[1]
    zer = np.zeros((h, w, 3), dtype=float)
    cimg = np.concatenate((img_fv, img, img_fv), axis=0)
    cimg2 = np.concatenate((zer, img_fh, zer), axis=0)
    cimg3 = np.concatenate((cimg2, cimg, cimg2), axis=1)
    zer = np.zeros((h, w, 1), dtype=float)
    cgt = np.concatenate((gt_fv, gt, gt_fv), axis=0)
    cgt2 = np.concatenate((zer, gt_fh, zer), axis=0)
    cgt3 = np.concatenate((cgt2, cgt, cgt2), axis=1)
    return cimg3, cgt3


def crop(img, x0, y0, w, h):
    return img[y0:y0 + h, x0:x0 + w, :]


def rotate_data(angle, img, gt):
    cimg, cgt = concatenate_image(img, gt)
    r_img = transform.rotate(cimg, angle)
    r_gt = transform.rotate(cgt, angle)
    h = img.shape[0]
    w = img.shape[1]
    img = crop(r_img, h, w, h, w)
    gt = crop(r_gt, h, w, h, w)
    return img, gt


def flip_img(horizontal_flip, img, gt_img):
    if horizontal_flip:
        img, gt_img = np.fliplr(img), np.fliplr(gt_img)
    return img, gt_img


def rotate_img(angle, img, gt_img):
    return transform.rotate(img, angle), transform.rotate(gt_img, angle)


def flip_and_rotate(horizontal_flip, angle, img, gt_img):
    img, gt = flip_img(horizontal_flip, img, gt_img)
    return rotate_data(angle, img, gt)


def resize_img(to_size, img):
    return transform.resize(img, (to_size, to_size))


def custom_image_generator(img_path, groundtruth_path, batch_size, random=True, preprocess=None):
    """
    Generator that returns images from the img_path folder. Useful to avoid loading all picture
    in memory.
    """
    imgs = np.array(os.listdir(img_path))
    i = 0
    while True:
        if random:
            batch_name = np.random.choice(a=imgs, size=batch_size)
        else:
            batch_name = imgs[i:i + batch_size]
            i = (i + batch_size) % len(imgs)

        if preprocess is not None:
            imgs_batch = np.array([preprocess(load_image(img_path + name)) for name in batch_name])
        else:
            imgs_batch = np.array([load_image(img_path + name) for name in batch_name])

        grnd_batch = np.array([load_image(groundtruth_path + name) for name in batch_name])

        yield imgs_batch, grnd_batch


def generate_files():
    """
    Function to reproduce data-augmentation we used for our last model.
    """
    imgs = os.listdir("./data/training/train/groundtruth")
    img_path = "data/training/train/images/"
    grnd_path = "./data/training/train/groundtruth/"

    tuples = [(name, load_image(img_path + name), load_image(grnd_path + name)) for name in imgs]

    func_list = [
        ("_flip", functools.partial(flip_and_rotate, True, 0)),
        ("_flip_90", functools.partial(flip_and_rotate, True, 90)),
        ("_flip_180", functools.partial(flip_and_rotate, True, 180)),
        ("_flip_270", functools.partial(flip_and_rotate, True, 270)),
        ("_90", functools.partial(flip_and_rotate, False, 90)),
        ("_180", functools.partial(flip_and_rotate, False, 180)),
        ("_270", functools.partial(flip_and_rotate, False, 270)),
        ("_flip_45", functools.partial(flip_and_rotate, True, 45)),
        ("_flip_135", functools.partial(flip_and_rotate, True, 135)),
        ("_45", functools.partial(flip_and_rotate, False, 45)),
        ("_135", functools.partial(flip_and_rotate, False, 135))
    ]

    for name, img, grnd in tuples:
        for suff, func in func_list:
            nimg, ngrnd = func(img, grnd)
            tmp_name = name.split('.')[0] + suff + "." + name.split('.')[1]

            im = Image.fromarray(plotting.grnd_to_img(ngrnd)[:, :, 1])
            im.save(grnd_path + tmp_name)

            im = Image.fromarray((nimg * 255).astype(np.uint8))
            im.save(img_path + tmp_name)
