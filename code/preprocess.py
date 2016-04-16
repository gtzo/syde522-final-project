# Preprocess histopath. images
from __future__ import division
from PIL import Image

import numpy as np
import glob
import pickle
import os

INPUT_DIMENSION = (308, 168)
OUTPUT_DIMENSION = INPUT_DIMENSION
INPUT_PIXELS = INPUT_DIMENSION[0] * INPUT_DIMENSION[1]

DATA_PATH = '../data/'

NUM_CLASSES = 20
IM_PER_CLASS = 48

RESIZE_WIDTH = 100 
wperc = RESIZE_WIDTH / float(INPUT_DIMENSION[0])
RESIZE_HEIGHT = int(float(INPUT_DIMENSION[1]) * float(wperc))

# Create labeled list of files
# Also downscale
def load_patches():
    labeled_patches = []
    for r, d, files in os.walk(DATA_PATH):
        for f in files:
            filename = DATA_PATH  + f
            im = Image.open(filename)

            # Downscale
            im = im.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.ANTIALIAS)
            im.save(f)

            label = f[0]
            labeled_patches.append((im, label))
            im.close()

    return labeled_patches

# Divide sets into tr_perc % training data
# and 100 - tr_perc % validation
def divide_sets(tr_perc, data):
    tr_size = int(tr_perc * IM_PER_CLASS)
    v_size = IM_PER_CLASS - tr_size

    training = []
    valid = []

    for c in range(0, NUM_CLASSES):
        starting_ind = IM_PER_CLASS * c
        for i in range(0, tr_size):
            ex = data[starting_ind + i] 
            training.append(ex)
        for j in range(tr_size, IM_PER_CLASS):
            ex = data[starting_ind + j]
            valid.append(ex)

    return training, valid

# Format of data point: data[0] == image, data[1] == label (string)
def grab_labels(data):
    l = []
    for i in data:
        if i[1] not in l:
            l.append(i[1])

    return l
