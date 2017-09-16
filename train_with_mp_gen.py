import cv2
import time
import glob
import multiprocessing as mp
import numpy as np
import keras

def process_img(image_file, idx=None):
    # print image_file
    img = cv2.imread(image_file)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (3,3))
    img = cv2.Canny(img, 10, 25)
    img = img.astype(np.float32) / 255
    if idx is None:
        return img
    else:
        return img, idx

def single_process_gen(image_files, batch_size=4):
    indices = np.arange(len(image_files))
    while True:
        for start in range(0, len(image_files), batch_size):
            end = min(len(image_files), start + batch_size)
            batch = []
            for idx in indices[start:end]:
                batch.append(process_img(image_files[idx]))
            batch = np.asarray(batch)
            yield batch, np.random.randint(0, 10, size=(len(batch), 1))

def multi_processes_gen(image_files, nproc, batch_size=4):
    def callback((img, idx)):
        batch[idx] = img

    indices = np.arange(len(image_files))
    image_files = np.asarray(image_files)
    while True:
        for start in range(0, len(image_files), batch_size):
            end = min(len(image_files), start + batch_size)
            batch = pool.map(process_img, image_files[indices[start:end]])
            # batch = [None] * (end - start + 1)
            # for idx in indices[start:end]:
            #     pool.apply_async(process_img, (image_files[idx],idx - start), callback=callback)
            # batch = np.asarray(batch)
            yield np.asarray(batch), np.random.randint(0, 10, size=(len(batch), 1))

if __name__ == '__main__':
    pool = mp.Pool(4)

    files = glob.glob('input/train_hq/*.jpg')
    gen = single_process_gen(files, batch_size=4)
    start = time.time()
    for _ in range(10):
        b = next(gen)
    print 'elapsed time', int(time.time() - start)

    gen = multi_processes_gen(files, batch_size=4, nproc=4)
    start = time.time()
    for _ in range(10):
        b = next(gen)
    print 'elapsed time', int(time.time() - start)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.sparse_categorical_crossentropy)
    model.fit_generator(generator=multi_processes_gen(files, nproc=4, batch_size=4),
                        steps_per_epoch=10,
                        epochs=10,
                        verbose=1)

    pool.close()
    pool.join()