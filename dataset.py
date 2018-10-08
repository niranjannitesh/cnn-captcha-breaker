import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

DATADIR = 'dataset'
CATEGORIES = ['2', '3', '4', '6', '7', '8', '9', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'P', 'Q', 'R', 'T', 'U', 'V', 'X', 'Y', 'Z']
IMG_SIZE = 40

# Show first image

# for category in CATEGORIES:
#   path = os.path.join(os.getcwd(), DATADIR, category)
#   for img in os.listdir(path):
#     img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#     img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     plt.imshow(img_array, cmap="gray")
#     plt.show()
#     break
#   break

training_data = []

def create_train_data():
  for category in CATEGORIES:
    path = os.path.join(os.getcwd(), DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
      try:
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([img_array, class_num])
      except:
        pass

create_train_data()

# shuffle data
import random
random.shuffle(training_data)

# testing show data

for sample in training_data[:5]:
  plt.imshow(sample[0], cmap="gray")
  plt.show()
  print(CATEGORIES[sample[1]])


# save_data
X = []
y = []

for features, label in training_data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open('X.pickle', "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
