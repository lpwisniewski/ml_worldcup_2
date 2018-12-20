import numpy as np
from tensorflow.python.keras import models

from losses import bce_dice_loss, dice_loss, f1_score


def load_model(model_save_path):
    """
    Load model from the given path.
    """
    return models.load_model(model_save_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                              'dice_loss': dice_loss,
                                                              'f1_score': f1_score})


def create_csv(grnd_list, csv_path, threshold):
    """
    Create csv from all the predicted groundtruth. It do the mean of each 16x16 patches then apply
    the threshold to now if this patches is mainly a road or not.
    """
    with open(csv_path, 'w') as f:
        f.write('id,prediction\n')
        for i, grnd in enumerate(grnd_list):
            for j in range(0, 38):
                for k in range(0, 38):
                    img_number = i + 1
                    label = np.mean(grnd[k * 16:k * 16 + 16, j * 16:j * 16 + 16]) > threshold
                    res = "{:03d}_{}_{},{}\n".format(img_number, j * 16, k * 16, int(label))
                    f.write(res)


def predict_test_img(img, model):
    """
    Function that takes a 608x608 image and returns it's predicted goundtruth by predicting
    4 corners of 400x400 then assembling them.
    """
    a = img[:400, :400]
    b = img[:400, 208:]
    c = img[208:, 208:]
    d = img[208:, :400]
    pred = model.predict(np.array(([a, b, c, d])))
    e = np.zeros((608, 608, 1))
    e[:400, :400] = pred[0]
    e[:400, 208:] = pred[1]
    e[208:, 208:] = pred[2]
    e[208:, :400] = pred[3]
    return e
