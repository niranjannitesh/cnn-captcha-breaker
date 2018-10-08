from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time


IMG_SIZE = 40
DATADIR = 'dataset'
model = load_model('4700_imgs_2_conv_2_dense_10epochs_10_val')
CATEGORIES = ['2', '3', '4', '6', '7', '8', '9', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'P', 'Q', 'R', 'T', 'U', 'V', 'X', 'Y', 'Z']

correct = 0
total = 0
wrong = 0


for category in CATEGORIES:
    path = os.path.join(os.getcwd(), DATADIR, category)
    print(f"---{category}---")
    i = 0
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
      img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
      img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
      p = model.predict([img_array])
      # print( CATEGORIES[np.asarray(np.argmax(p, axis=1))[0]] )
      if (CATEGORIES[np.asarray(np.argmax(p, axis=1))[0]] == category):
        correct += 1
      else:
        wrong += 1
      # print(p)
      i += 1
      total += 1
      # if (i == 100):
      #   break

print()
print(correct * 100 / total, "%", "is correct")
print(f"{wrong} wrong guess")
print(f"{correct} correct guess")
print(f"{total} total letters")
