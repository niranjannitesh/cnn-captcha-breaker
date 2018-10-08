import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.backend import argmax
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

NAME = "4700_imgs_2_conv_2_dense_10epochs_10_val"

# load data
X = pickle.load(open('X.pickle', "rb"))
y = to_categorical(pickle.load(open('y.pickle', "rb")))

CATEGORIES = ['2', '3', '4', '6', '7', '8', '9', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'P', 'Q', 'R', 'T', 'U', 'V', 'X', 'Y', 'Z']
IMG_SIZE = 40
DATADIR = 'dataset'

# normalize data
X = X / 255.0

# CNN model

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(50, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(100))
model.add(Activation("tanh"))


model.add(Dense(27))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])


print(f"Saving current model as {NAME}")
model.save(f'{NAME}')
