# ====
# Keras autoencoder sample
# ===
from __future__ import division

# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
import keras

# Other
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp

VALIDATION_SPLIT = 0.2
EPOCHS = 20

np.random.seed(1)

print "Loading data."
patches, labels = pp.load_patches()
#training, validation, t_labels, v_labels = pp.divide_sets(TRAINING_RATIO, patches, labels)

print "Building network."
model = Sequential() # Linear stack of layers
model.add(Dense(1000, input_dim=pp.RESIZE_PIXELS, init='uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

"""
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
"""

model.add(Dense(pp.NUM_CLASSES, init='uniform'))
model.add(Activation('softmax'))
model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

history = model.fit(patches, labels, nb_epoch=EPOCHS, batch_size=16, show_accuracy=True, verbose=True, validation_split=VALIDATION_SPLIT)
print 'Done baseline training.'
