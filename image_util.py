from PIL import Image
import cv2, os
import numpy as np
from config import config as cfg
from preprocess_img import *


def imread(imgpath):
    img = cv2.imread(imgpath)
    return img[:,:,::-1]

def extend_img(img):
    img = random_hue(img)
    img = random_saturation(img)
    img = random_brightness(img)
    return img

def gen_images_batch(imgpaths):
    img_batch = []
    for imgpath in imgpaths:
        img = imread(imgpath)
        img = cv2.resize(img, (cfg.imagesize, cfg.imagesize), interpolation=cv2.INTER_LINEAR)
        if True:
            img = extend_img(img)
        img_batch.append(img)
    return np.stack(img_batch, 0)

def labelread(labelpath):
    label = []
    f = open(labelpath, "r")
    while True:
        lcbox = f.readline().strip()
        if len(lcbox) == 0:
            break
        lcbox = lcbox.split(" ")
        lcbox = list(map(np.float32, lcbox))
        label.append(np.array(lcbox))
    f.close()
    return np.stack(label, 0)


def gen_labels_batch(imgpaths):
    labels_batch = []
    label_num = []
    for imgpath in imgpaths:
        label_array = np.zeros(shape=[cfg.targets_per_image, 5], dtype=np.float32)
        labelpath = imgpath.split(".")[0] + ".txt"
        # labelpath = labelpath.replace("Tank_JPEG", "Tank")
        label = labelread(labelpath)
        num = label.shape[0]
        if num>cfg.targets_per_image:
            label_array[0:cfg.targets_per_image, :] = label[0:cfg.targets_per_image,:]
        else:
            label_array[0:num, :] = label
        labels_batch.append(label_array)
        label_num.append(num)
    return np.stack(labels_batch, 0), np.array(label_num)






