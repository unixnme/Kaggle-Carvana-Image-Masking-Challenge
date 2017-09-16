import cv2
import time
import glob
import multiprocessing as mp

def process_img(image_file):
    if '_canny.jpg' not in image_file:
        print 'processing', image_file
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img, (3,3))
        img = cv2.Canny(img, 10, 25)
        cv2.imwrite(image_file[:-4] + '_canny.jpg', img)

def single_process(image_files):
    start = time.time()
    for image_file in image_files:
        process_img(image_file)
    print 'single process elapsed time:', int(time.time() - start)

def multi_processes(image_files, nproc=2):
    start = time.time()
    pool = mp.Pool(nproc)
    for image_file in image_files:
        pool.apply_async(process_img, (image_file,))
    pool.close()
    pool.join()
    print nproc, 'processes elapsed time:', int(time.time() - start)

if __name__ == '__main__':
    files = glob.glob('input/train_hq/*.jpg')
    single_process(files)
    multi_processes(files, nproc=2)
    multi_processes(files, nproc=4)
