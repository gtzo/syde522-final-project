# ====
# Keras autoencoder sample
# ===
from __future__ import division

# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import keras

# Other
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp

TRAINING_RATIO = 0.7
EPOCHS = 50

np.random.seed(1)

print "Loading data."
patches, labels = pp.load_patches()
#training, validation, t_labels, v_labels = pp.divide_sets(TRAINING_RATIO, patches, labels)

def build_baseline_network():
    model = Sequential() # Linear stack of layers

    model.add(Dense(128, input_dim=pp.RESIZE_PIXELS))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Classif. layer
    model.add(Dense(pp.NUM_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print 'Starting baseline training.'

    model.fit(patches, labels, nb_epoch=EPOCHS, batch_size=100, show_accuracy=True, verbose=True, validation_split=0.1)

    print 'Done baseline training.'

    return model

print "Building network."
net = build_baseline_network()
