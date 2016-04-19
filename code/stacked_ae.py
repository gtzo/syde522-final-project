# ====
# HISTOPATHOLOGY AUTOENCODER
# SYDE 522: Machine Intelligence Final Project
# George Tzoganakis
# ===

from __future__ import division
from utils import load_mnist, plot_layers, plot_images

import theanets
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp

np.random.seed(1)

TRAINING_RATIO = 0.7
MIDDLE_HUNITS = 100 

# ===
# Loading data
# ===
print "Loading data."
patches, labels = pp.load_patches(False)
labels = labels.astype('int32')
training, validation, t_labels, v_labels = pp.divide_sets(TRAINING_RATIO, patches, labels)

# ===
# Creating network
# ===
print "Building network."
net = theanets.Autoencoder(
    layers=(pp.RESIZE_PIXELS, 750, MIDDLE_HUNITS, ('tied', 750), ('tied', pp.RESIZE_PIXELS)),
)

# ===
# Training network
# ===
print "Training network."
net.train([training], [validation],
          algo='layerwise',
          patience=1,
          min_improvement=0.05,
          train_batches=100)

# Fine tuning
#net.train(training, validation, min_improvment=0.01, train_batches=100)

# ===
# Training softmax classifier 
# ===
classifier = theanets.Classifier([MIDDLE_HUNITS, pp.NUM_CLASSES])
training_activations = net.encode(training)
data = np.asarray([training_activations, t_labels])
classifier.train([training_activations, t_labels], patience=1, min_improvement=0.01)

# ===
# Validation
# ===
valid_activations = net.encode(validation)
predictions = classifier.predict(valid_activations)
print v_labels
print "Classifications:"
print predictions

# ===
# Storing weights and plotting relevant results
# ===
net.save('autoencoder_state_1200-500')
classifier.save('classifier_state_1200-500')

""" 
plot_layers([net.find(i, 'w') for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()
"""

valid = validation[:100]
plot_images(valid, 121, 'Sample data')
plot_images(net.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()
