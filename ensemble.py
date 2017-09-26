import keras
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, sparse_cce_dice_loss
from model.u_net import get_unet_256

batch_size = 1
df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

def valid_generator(rows, cols):
    print 'loading', len(ids_valid_split), 'samples'
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread('input/train_hq/{}.jpg'.format(id))
                img = cv2.resize(img, (cols, rows))
                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (cols, rows))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def test_model1():
    rows = 1024
    cols = 1024
    model = load_model('weights/unet_1024_9960.hdf5', compile=False)
    model.compile(optimizer=SGD(), loss=bce_dice_loss, metrics=[dice_coeff])
    gen = valid_generator(rows, cols)
    accuracy = 0.
    for _ in range(1018):
        x,y = next(gen)
        loss, acc = model.evaluate(x,y)
        accuracy += acc
    print 'average accuracy is', accuracy / 1018

def test_model2():

if __name__ == '__main__':
    test_model1()
