# ====
# HISTOPATHOLOGY AUTOENCODER
# SYDE 522: Machine Intelligence Final Project
# George Tzoganakis
# ===

from __future__ import division
from utils import load_mnist, plot_layers, plot_images
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold

import csv
import theanets
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp

np.random.seed(1)

TRAINING_RATIO = 0.7
MIDDLE_HUNITS = 256 

# ===
# Loading data
# ===
print "Loading data."
patches, labels = pp.load_patches(False)
labels = labels.astype('int32')
training, validation, t_labels, v_labels, class_means = pp.divide_sets(TRAINING_RATIO, patches, labels)

# ===
# Creating network
# ===
print "Building network."
net = theanets.Autoencoder(
    layers=(pp.RESIZE_PIXELS, 1500, 900, 500, MIDDLE_HUNITS, ('tied', 500), ('tied', 500), ('tied', 1500), ('tied', pp.RESIZE_PIXELS)),
)

# ===
# Training network
# ===
print "Training network."
drp = [0.0, 0.1, 0.3, 0.5]
for j in drp:
    net.train(training, validation, hidden_dropout=j,
              algo='layerwise',
              patience=1,
              min_improvement=0.05,
              train_batches=100)
    ti = 'ae_1500-900-500-256-dropout' + str(j)
    net.save(ti)

"""
for j in drp:
    net.train(training, validation, hidden_l1=j,
              algo='layerwise',
              patience=1,
              min_improvement=0.05,
              train_batches=100)
    ti = 'ae_1500-900-500-256-sparse' + str(j)
    net.save(ti)
"""

# Fine tuning
#net.train(training, validation, min_improvment=0.01, train_batches=100)

"""
# ===
# Training softmax classifier 
# ===
classifier = theanets.Classifier([MIDDLE_HUNITS, pp.NUM_CLASSES])
training_activations = net.encode(training)
data = np.asarray([training_activations, t_labels])
classifier.train([training_activations, t_labels], patience=1, min_improvement=0.01)
"""

"""
# Load network
net = theanets.Network.load('autoencoder_state_1000-750-300')
training_activations = net.encode(training)
valid_activations = net.encode(validation)
"""

"""
# ===
# Validation using softmax
# ===
valid_activations = net.encode(validation)
predictions = classifier.predict(valid_activations)

compare = predictions == v_labels
success = 0
for i in compare:
    if i:
        success += 1

success_rate = success / len(v_labels)
"""

# ===
# Storing weights and plotting relevant results
# ===
#net.save('autoencoder_state_1000-750-300')
#classifier.save('classifier_state_1200-500')

"""
# ==
# Feature sel
# ==
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(training_activations)
"""

"""
# ===
# Classification using KNN
# ===
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(training_activations, t_labels)
k_score = neigh.score(valid_activations, v_labels)

# ===
# Classification using tree 
# ===
tr = tree.DecisionTreeClassifier()
tr = tr.fit(training_activations, t_labels)
tr_score = tr.score(valid_activations, v_labels)

# ===
# Classification using tree 
# ===
ba = GaussianNB()
y_pr = ba.fit(training_activations, t_labels)
b_score = ba.score(valid_activations, v_labels)

# ===
# Classification using naive bayes 
# ===
ba = GaussianNB()
y_pr = ba.fit(training_activations, t_labels)
b_score = ba.score(valid_activations, v_labels)

# ===
# Classification using random forest  
# ===
ba = RandomForestClassifier(n_estimators=100)
y_pr = ba.fit(training_activations, t_labels)
rf_score = ba.score(valid_activations, v_labels)

# ===
# Results
# ===
results = dict()
results['knn'] = k_score
results['dt'] = tr_score
results['bayes'] = b_score

print '=== Results ==='
print 'KNN .: ', k_score
print 'Decision tree .: ', tr_score
print 'Naive bayes .: ', b_score
print 'Random forest .: ', rf_score
"""

"""
plot_layers([net.find(i, 'w') for i in (1, 2, 3)], tied_weights=True)
plt.tight_layout()
plt.show()
"""

"""
valid = validation[:100]
plot_images(valid, 121, 'Sample data')
plot_images(net.predict(valid), 122, 'Reconstructed data')
plt.tight_layout()
plt.show()
"""
