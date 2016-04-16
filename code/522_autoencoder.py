# ===============
# SYDE 522: Machine Intelligence
# Assignment 2
# ===============

from __future__ import division
from PIL import Image

import matplotlib.pyplot as plt
import time
import pickle
import theanets
import glob
import numpy as np

PATCH_SIZE = 32
vlayer_size = PATCH_SIZE**2
hlayer_sizes = [
                vlayer_size/16,
                vlayer_size/8,
                vlayer_size/4,
                vlayer_size/2]
#hlayer_sizes = [100]
training_times = []
ae_errors = []
scores = []

# Preprocess patches 
def load_patches():
    patches_list = glob.glob('*.png')
    patches = [] 

    for i in patches_list:
        im = Image.open(i).convert('L')
        im = im.resize((PATCH_SIZE, PATCH_SIZE), Image.ANTIALIAS)
        arr = np.asarray(im)
        arr = arr.reshape(-1,).astype('f')
        patches.append(arr)

    patches = np.asarray(patches)
    return patches / 255

# Load and partition data
data = load_patches()
training = data[:40]
validation = data[41:56] 

#  Stochastic corruption step for denoising autoencoders
for q in training:
    zeros = np.random.randint(0, PATCH_SIZE**2 / 2) # choose an arbitrary number of zeroed inputs
    for i in range(0, zeros):
        q[np.random.randint(0, len(q)-1)] = 0 # set them to zero

# Train and examine each autoencoder size
for i, s in enumerate(hlayer_sizes):
    net = theanets.Autoencoder([PATCH_SIZE**2, s, PATCH_SIZE**2])

    start = time.time()
    net.train(training)
    training_times.append(time.time() - start) # Record training time

    # Generate a metric of success
    prediction = net.predict(validation)
    local_errors = []
    for j, p in enumerate(prediction):
        error = (np.sum(np.subtract(p, validation[j])**2)) / PATCH_SIZE**2
        local_errors.append(error)
    error = np.average(local_errors)

    # Save the trained network to avoid retraining
    title = 'autoencoder_trained' + str(i) + '.pkl'
    autoencoder = net.save(title)

    # Store performance data
    ae_errors.append(error)
    scores.append(net.score(validation))
    
# Present results
print "Training times: ", training_times
print "RMSEs, input vs output: ", ae_errors
print "R^2 scores, input vs output: ", scores

plt.figure(1)
plt.plot(hlayer_sizes, training_times)
plt.title('Hidden layer sizes vs. training times')
plt.xlabel('hidden layer sizes')
plt.ylabel('training times, s')

plt.figure(2)
plt.plot(hlayer_sizes, ae_errors)
plt.title('Hidden layer sizes vs. MSE')
plt.xlabel('hidden layer sizes')
plt.ylabel('MSE')

plt.figure(3)
plt.plot(hlayer_sizes, scores)
plt.title('Hidden layer sizes vs. R^2 coefficients')
plt.xlabel('hidden layer sizes')
plt.ylabel('R^2')

plt.show() 
