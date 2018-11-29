import os

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


def load_image(infilename):
    data = mpimg.imread(infilename)
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
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
    print("Loading " + str(n) + " ground truth")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

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
        tr_img, grd_img = tf.image.flip_left_right(img), tf.image.flip_left_right(gt_img)
    else:
        tr_img, grd_img = tf.image.flip_up_down(img), tf.image.flip_up_down(gt_img)
    return tr_img, grd_img


def rotate_img(rotate_plus, img, gt_img):
    if rotate_plus:
        tr_img, grd_img = tf.image.rot90(img, k=1), tf.image.rot90(gt_img, k=1)
    else:
        tr_img, grd_img = tf.image.rot90(img, k=3), tf.image.rot90(gt_img, k=3)
    return tr_img, grd_img


def get_baseline_dataset(imgs,
                         labels,
                         list_improve_func=[],
                         threads=5,
                         batch_size=10,
                         shuffle=True):
    """
    Get a full dataset with data improved.
    :param imgs: images in tensor form
    :param labels: groundtruth in the tensor form
    :param list_improve_func: List of function you want to apply on your data to improve it
    :param threads: Number of thread you want to create datasets and apply improve functions
    :return: A repeated dataset, batched and improved with given function.
    """

    num_x = len(imgs)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

    # Apply all improvement function to our dataset and store results
    datasets = []
    for i in range(0, len(list_improve_func)):
        datasets.append(dataset.map(list_improve_func[i], num_parallel_calls=threads))

    # Concatenate all datasets in one
    for i in range(0, len(datasets)):
        dataset = dataset.concatenate(datasets[i])

    if shuffle:
        dataset = dataset.shuffle(num_x * (len(list_improve_func) + 1))

    # It's necessary to repeat our data for all epochs
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()

    num_batch = 0
    while True:
        try:
            sess.run(next_element)
            num_batch += 1
        except tf.errors.OutOfRangeError:
            break
    dataset = dataset.repeat().batch(batch_size)
    return dataset
