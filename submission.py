import numpy as np
from tensorflow.python.keras import models

from losses import bce_dice_loss, dice_loss, f1_score


def load_model(model_save_path):
    return models.load_model(model_save_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                              'dice_loss': dice_loss,
                                                              'f1_score': f1_score})


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
