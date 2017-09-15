import cv2
import glob
import multiprocessing as mp

def process_image(file):
    print file
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    img_canny = cv2.Canny(gray, 10, 25)
    cv2.imwrite(file[:-4] + '_canny.jpg', img_canny)

def single_process(files):
    for file in files:
        if file[-9:] == 'canny.jpg':
            continue
        process_image(file)

def multi_processes(files):
    pool = mp.Pool(3)
    for file in files:
        if file[-9:] == 'canny.jpg':
            continue
        pool.apply_async(process_image, args=(file,))
    pool.close()
    pool.join()

if __name__ == '__main__':
    files = glob.glob("input/train_hq/*.jpg")
    multi_processes(files)
    #single_process(files)