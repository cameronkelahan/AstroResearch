# name: multiDataPreprocessing.py
# purpose: Loading of data from .txt to arrays and rearrangement of said arrays into training set and training labels

NB_CLASSES=10
import numpy as np
from keras.utils import np_utils

def processTestData(X, y):

    # X preprocessing goes here -- students optionally complete

    # y preprocessing goes here.  y_test becomes a ohe
    y_ohe = np_utils.to_categorical (y, NB_CLASSES)
    return X, y_ohe