import glob
import os

import cv2
import numpy as np


def read_img(path, wide, height):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    cate.sort(key=lambda arr: (arr[:24], int(arr[24:])))
    images = []
    labels = []

    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s'%(im))
            img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (wide, height))
            images.append(np.array(img).flatten('F'))
            labels.append(idx + 1)

    return np.asarray(images, np.float32), np.asarray(labels, np.int32)

