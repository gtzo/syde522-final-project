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
labels_list = pp.grab_labels(labels)
training, validation, t_labels, v_labels = pp.divide_sets(TRAINING_RATIO, patches, labels)

def build_baseline_network():
    model = Sequential() # Linear stack of layers

    # output size = 1000
    model.add(Dense(1000, input_dim=pp.RESIZE_PIXELS, activation='tanh'))
    model.add(Dropout(0.5))

    # Classif. layer
    model.add(Dense(len(labels)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0)

    print 'Starting baseline training.'
    model.fit(training, t_labels, nb_epoch=EPOCHS, batch_size=10, show_accuracy=True, verbose=True, callbacks=[early_stopping])
    print 'Done baseline training.'

    return model

print "Building network."
net = build_baseline_network()
score = net.evaluate(validation, v_labels, show_accuracy=True, verbose=False)
print 'Accuracy: ' + str(score)
