# ====
# Keras autoencoder sample
# ===
from __future__ import division

# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# Other
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp

TRAINING_RATIO = 0.7
EPOCHS = 50

np.random.seed(1)
random.seed(1)

patches = pp.load_patches()
labels = pp.grab_labels(patches)
training, validation = pp.divide_sets(TRAINING_RATIO, patches)

def build_network():
    model = Sequential() # Linear stack of layers

    model.add(Dense(16, input_dim=np.prod(INPUT_DIMENSION)) # Input (*, INPUT_DIM), output (*, 16)
    model.add(Dropout(0.5))

    # Classif. layer
    model.add(Dense(len(labels)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model
