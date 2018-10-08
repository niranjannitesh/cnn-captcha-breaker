import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 40

def load_captcha_file(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = img[:, 0:150]
    return img

def cut_captcha(img):
    char_imgs = []
    for i in range(0, 4):
        char = img[0:40, 37 * i: 37 * (i + 1)]
        char = cv2.resize(char, (IMG_SIZE, IMG_SIZE))
        # plt.imshow(char, cmap="gray")
        # plt.show()
        char_imgs.append(char)
    return char_imgs
