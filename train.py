import cv2
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split
import os
import sys
import params
from model.u_net import leaky, relu, prelu, elu
from model.low_res import create_model
from model.losses import bce_dice_loss, dice_coeff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

def step_decay(epoch):
    lr = learning_rate * np.power(0.5, epoch/float(half_life))
    print 'learning rate =', lr
    return lr

def preprocess_input(x_batch):
    x_batch[..., 0] -= 103.939
    x_batch[..., 1] -= 116.779
    x_batch[..., 2] -= 123.68
    return x_batch

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,), flags=cv2.INTER_NEAREST)
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomCrop(img, mask, crop_size):
    rows, cols = crop_size
    x0 = 0
    y0 = 0
    if cols < img.shape[1]:
        x0 = np.random.randint(img.shape[1] - cols)
    if rows < img.shape[0]:
        y0 = np.random.randint(img.shape[0] - rows)
    return img[y0:y0+rows, x0:x0+cols, :], mask[y0:y0+rows, x0:x0+cols]

def train_generator(save_to_ram=False):
    indices = np.array(ids_train_split.index)
    cache = {}
    while True:
        np.random.shuffle(indices)
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[indices[start:end]]
            for id in ids_train_batch.values:
                if save_to_ram is True and cache.has_key(id):
                    img, mask = cache[id]
                else:
                    img = cv2.imread('input/train_hq/{}.jpg'.format(id))
                    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id))
                    img = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_NEAREST)
                    # mask color to gray
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if save_to_ram is True:
                    cache[id] = (img, mask)

                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-5, 5),
                                                   u=1)
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255 - 0.5 + input_mean
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def valid_generator(save_to_ram=False):
    cache = {}
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                if save_to_ram is True and cache.has_key(id):
                    img, mask = cache[id]
                else:
                    img = cv2.imread('input/train_hq/{}.jpg'.format(id))
                    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id))
                    img = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_NEAREST)
                    # mask color to gray
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if save_to_ram is True:
                    cache[id] = (img, mask)

                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255 - 0.5 + input_mean
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


if __name__ == '__main__':

    epochs = 1000
    batch_size = 10
    rows, cols = 256, 256
    learning_rate = 1e-3
    input_mean = .5
    decay = 0.5
    offset = 11

    df_train = pd.read_csv('input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    steps_per_epoch = np.ceil(float(len(ids_train_split)) / float(batch_size))
    validation_steps = np.ceil(float(len(ids_valid_split)) / float(batch_size))
    # steps_per_epoch, validation_steps = 10, 10

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    activations = [relu, relu, elu, elu, leaky, leaky, prelu, prelu]
    BNs =         [False, True, False, True, False, True, False, True]

    for idx in range(6):
        name = 'exp' + str(idx + offset)
        with open('nohup.out.' + name, 'w') as f:
            sys.stdout = f
            filepath = 'weights/' + name + '_model.h5'

            model = create_model(shape=(rows, cols, 3),
                                 num_blocks=3,
                                 kernel=3,
                                 filter=4,
                                 dilation=1,
                                 regularizer=None,
                                 activation=activations[idx],
                                 BN=BNs[idx],
                                 pooling='max')
            # model.load_weights(filepath, by_name=True)
            model.compile(optimizer=RMSprop(learning_rate), loss=bce_dice_loss, metrics=[dice_coeff])

            callbacks = [ModelCheckpoint(monitor='val_loss',
                                         filepath=filepath,
                                         verbose=True,
                                         save_best_only=True,
                                         save_weights_only=False),
                         ReduceLROnPlateau(monitor='val_loss',
                                           factor=decay,
                                           patience=3,
                                           verbose=1,
                                           epsilon=1e-4,
                                           mode='min',
                                           min_lr=1e-5),
                         EarlyStopping(monitor='val_loss',
                                           patience=5,
                                           verbose=1,
                                           mode='min',
                                           min_delta=1e-5)]

            history = model.fit_generator(generator=train_generator(True),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=valid_generator(True),
                                validation_steps=validation_steps)
            with open(name + '_history.p', 'wb') as f:
                pickle.dump(history.history, f)

            print(history.history.keys())
            #  "Accuracy"
            fig = plt.figure()
            plt.semilogy(history.history['dice_coeff'])
            plt.semilogy(history.history['val_dice_coeff'])
            plt.title('model accuracy')
            plt.ylabel('dice coefficient')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            fig.savefig(name + '_acc.png')
            # "Loss"
            fig = plt.figure()
            plt.semilogy(history.history['loss'])
            plt.semilogy(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            fig.savefig(name + '_loss.png')
            # "LearningRate"
            fig = plt.figure()
            plt.plot(history.history['lr'])
            plt.title('learning rate')
            plt.ylabel('learning rate')
            plt.xlabel('epoch')
            fig.savefig(name + '_lr.png')

            del model
