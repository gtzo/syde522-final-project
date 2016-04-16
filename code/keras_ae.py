# ====
# Keras autoencoder sample
# ===
from __future__ import division
from PIL import Image

# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# Other
import matplotlib.pyplot as plt
import time
import pickle
import theanets
import glob
import numpy as np

INPUT_DIMENSION = 32
OUTPUT_DIMENSION = INPUT_DIMENSION # Symmetrical

def build_autoencoder():
    model = Sequential() # Linear stack of layers

    model.add(Dense(16, input_dim=INPUT_DIMENSION)) # Input (*, INPUT_DIM), output (*, 16)
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(OUTPUT_DIMENSION))
    model.add(Activation('softmax'))
