import pandas as pd
from IPython.display import display
import numpy as np
import seaborn as sns
# Plotting 
from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline

# Train, valid and test sets
from sklearn.model_selection import train_test_split

# Evaluation
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Keras imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential, load_model 
from tensorflow.keras.layers import Bidirectional, LayerNormalization, LeakyReLU, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Flatten, BatchNormalization, Permute
from tensorflow.keras.optimizers import schedules, Adam, SGD

# Local imports 
import processing
import dryad_process
from evaluation import f1_score_all, er_all, get_score, sed_eval_scores, plotPredTruth, plotPred, plotTruth

###########################
def preprocess(x, y):
    x = np.array([np.array(val, dtype='float') for val in x])
    y = np.array([np.array(val, dtype='float') for val in y])
    
    # Reshape sets: (length of set, frames, mels, 1)
    x = x.reshape(len(x), x[0].shape[0], x[0].shape[1], 1)
    
    return x, y

###########################
# Parameters
n_files = 1
polyphony = 10
threshold = 0.5

# Load model
model = load_model('dryad_day1_o10.h5')

# Generate data
spectrograms, annotations = dryad_process.generateDataset(n_files, polyphony)
spectrograms, annotations = preprocess(spectrograms, annotations)
in_shape = spectrograms[0].shape
out_shape = annotations[0].shape

# Make prediction
cnt = 0
for elem in spectrograms:
    elem = elem.reshape(1, in_shape[-3], in_shape[-2], in_shape[-1])
    prediction = model.predict(elem)

    predNoThres = prediction.reshape(out_shape[-2], out_shape[-1])    
    predNoThres = np.array(predNoThres)
    predBinary = processing.output_to_binary(prediction, threshold)
    predBinary = predBinary.reshape(out_shape[-2], out_shape[-1])

    ov = processing.countOverlap(annotations[cnt].T)

    # Score
    f1, er = get_score(predBinary.T, annotations[cnt].T)
    print("Polyphony: ", ov)
    print("F1: ", f1)
    print("ER: ", er)

    # Plot 
    plotPred(predBinary.T, dryad_process.speciesList)
    plotTruth(annotations[cnt].T, dryad_process.speciesList)

    cnt += 1