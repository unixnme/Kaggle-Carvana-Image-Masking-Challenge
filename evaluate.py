from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, sparse_cce_dice_loss
from keras.models import load_model
from keras.optimizers import SGD
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

cols, rows = 1024, 1024
batch_size = 10

def valid_generator(save_to_ram=False):
    cache = {}
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                if save_to_ram is True and cache.has_hey(id):
                    img, mask = cache[id]
                else:
                    img = cv2.imread('input/train_hq/{}.jpg'.format(id))
                    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (cols, rows))
                    mask = cv2.resize(mask, (cols, rows), cv2.INTER_NEAREST)
                if save_to_ram is True:
                    cache[id] = (img, mask)

                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            #x_batch = preprocess_input(x_batch)
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

model = load_model('weights/unet_1024_9960.hdf5', compile=False)
model.compile(optimizer=SGD(), loss=bce_dice_loss, metrics=[dice_coeff])

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

gen = valid_generator()
avg_acc = 0.
for _ in range(1018/batch_size):
    x,y = next(gen)
    loss, acc = model.evaluate(x,y, verbose=0)
    print acc
    avg_acc += acc
print avg_acc / (1018/batch_size)


