import cv2
import time
import glob
import multiprocessing.pool as mp

def process_img(image_file):
    if '_canny.jpg' not in image_file:
        print 'processing', image_file
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img, (3,3))
        img = cv2.Canny(img, 10, 25)
        cv2.imwrite(image_file[:-4] + '_canny.jpg', img)

def single_thread(image_files):
    start = time.time()
    for image_file in image_files:
        process_img(image_file)
    print 'single thread elapsed time:', int(time.time() - start)

def multi_threads(image_files, nproc=2):
    start = time.time()
    pool = mp.ThreadPool(nproc)
    for image_file in image_files:
        pool.apply_async(process_img, (image_file,))
    pool.close()
    pool.join()
    print nproc, 'threads elapsed time:', int(time.time() - start)

if __name__ == '__main__':
    files = glob.glob('input/train_hq/*.jpg')
    #single_thread(files)
    #multi_threads(files, nproc=2)
    multi_threads(files, nproc=4)
