from tensorflow.python.keras.models import load_model
import tensorflow as tf
import numpy as np
from img import IMG_SIZE
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORIES = ['2', '3', '4', '6', '7', '8', '9', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'P', 'Q', 'R', 'T', 'U', 'V', 'X', 'Y', 'Z']

def init_model():
    model = load_model('model_4000')
    return model

def get_captcha_from_array(model, array):
    word_text = ""
    for char in array:
        word = np.array(char).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        pred = model.predict([word])
        word_text += CATEGORIES[np.asarray(np.argmax(pred, axis=1))[0]]
    return word_text
