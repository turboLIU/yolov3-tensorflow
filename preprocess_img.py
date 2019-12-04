import cv2
import numpy as np

def random_flip(img, c):  # n==1:# flip
    img = cv2.flip(img, c)  # 1:水平翻转； 0：垂直翻转； -1：水平翻转+垂直翻转
    return img


def random_brightness(img):  # n==2: # brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] * np.random.uniform(0.5, 1.2, 1)
    v = np.where(v > 180, 180, v)
    hsv[:, :, 2] = v
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_hue(img):  # n==3: # hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0] * np.random.uniform(0.5, 1.2, 1)
    h = np.where(h > 180, 180, h)
    hsv[:, :, 0] = h
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_saturation(img):  # n==4: # saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1] * np.random.random()
    hsv[:, :, 1] = s
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def random_contrast(img):  # n==5: # contrast
    img_c = []
    for i in range(img.shape[2]):
        tmp = (img[:, :, i] - np.mean(img[:, :, i])) * np.random.uniform(0.8, 1.2, 1) + np.mean((img[:, :, i]))
        img_c.append(tmp)
    img = np.stack(img_c, 2)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.array(img * 255, np.uint8)
    return img