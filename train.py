import cv2
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop
from sklearn.model_selection import train_test_split
import os
import params
from model.u_net import leaky, relu

filepath= 'weights/best_weights_vgg16_crop_preprocess_normalize.hdf5'
rows = params.rows
cols = params.cols
epochs = params.max_epochs
batch_size = params.batch_size
learning_rate = 1e-2
half_life = 160
crop_size = 256
model = params.model_factory(input_shape=(None, None, 3),
        num_classes=1,
        optimizer=
        SGD(lr=1e-4, momentum=0.95, nesterov=True),
        activation=relu,
        regularizer=keras.regularizers.l2(1e-4))

if os.path.isfile(filepath):
    model.load_weights(filepath, by_name=True)

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

def step_decay(epoch):
    lr = learning_rate * np.power(0.5, epoch/float(half_life))
    print 'learning rate =', lr
    return lr

def preprocess_input(x_batch):
    x_batch[..., 0] -= 103.939
    x_batch[..., 1] -= 116.779
    x_batch[..., 2] -= 123.68
    return x_batch / 255

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
                                        0,))
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
    x0 = np.random.randint(img.shape[1] - crop_size)
    y0 = np.random.randint(img.shape[0] - crop_size)
    return img[y0:y0+crop_size, x0:x0+crop_size, :], mask[y0:y0+crop_size, x0:x0+crop_size]

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
                    mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (cols, rows))
                    mask = cv2.resize(mask, (cols, rows), cv2.INTER_NEAREST)
                if save_to_ram is True:
                    cache[id] = (img, mask)

                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-45, 45),
                                                   u=1)
                img, mask = randomCrop(img, mask, crop_size)
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32)
            x_batch = preprocess_input(x_batch)
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
            x_batch = np.array(x_batch, np.float32)
            x_batch = preprocess_input(x_batch)
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


if __name__ == '__main__':
    callbacks = [ModelCheckpoint(monitor='val_loss',
                                 filepath=filepath,
                                 verbose=True,
                                 save_best_only=True,
                                 save_weights_only=False),
		 ReduceLROnPlateau(monitor='val_loss', 
                                   factor=0.5,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-5),
                 LearningRateScheduler(step_decay),
                 TensorBoard(log_dir='logs')]

    model.fit_generator(generator=train_generator(False),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(False),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
