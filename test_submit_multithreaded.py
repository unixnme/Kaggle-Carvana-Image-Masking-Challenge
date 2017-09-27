import cv2
import keras
import numpy as np
import pandas as pd
import threading
from multiprocessing import Queue
import tensorflow as tf
from tqdm import tqdm
from model.u_net import leaky, relu
from keras.optimizers import SGD
from keras.models import load_model
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, sparse_cce_dice_loss

import params

rows, cols = 1024, 1024
batch_size = 10
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = load_model('weights/unet_1024_9960.hdf5', compile=False)
model.compile(optimizer=SGD(), loss=bce_dice_loss, metrics=[dice_coeff])

df_test = pd.read_csv('input/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


rles = []

graph = tf.get_default_graph()

q_size = 4


def data_loader(q, ):
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            img = cv2.imread('input/test_hq/{}.jpg'.format(id))
            img = cv2.resize(img, (cols, rows))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put(x_batch)


def predictor(q, ):
    for i in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = q.get()
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for pred in preds:
            prob = cv2.resize(pred, (orig_width, orig_height))
            mask = prob > threshold
            rle = run_length_encode(mask)
            rles.append(rle)


q = Queue(maxsize=q_size)
t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))
print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
t1.start()
t2.start()
# Wait for both threads to finish
t1.join()
t2.join()

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
