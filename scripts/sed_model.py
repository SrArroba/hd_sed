import pandas as pd
import numpy as np

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
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Flatten

# Local imports 
import processing

########################################################################
#                            FUNCTIONS                                 # 
########################################################################
def generate_sets():
    # Generate entire dataset
    n_files = 100
    polyphony = 3 
    train_x, train_y = processing.generateDataset(n_files, polyphony)

    # Separate into validation and training sets
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

    # Pre-process 
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test, y_train, y_test):
    # Reshape sets
    X_train = X_train.reshape(len(X_train),n_mels,time_stamps,1)
    X_test = X_test.reshape(len(X_test),n_mels,time_stamps,1)
    # Change output to categorical
    print(y_train[0])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def create_model():
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(n_mels,time_stamps,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))
    print (model.summary())

    #compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
########################################################################
#                                 MAIN                                 # 
########################################################################

# Main values
n_classes = 87
time_stamps = 430
n_mels = 128


X_train, X_test, y_train, y_test = generate_sets()
model = create_model()

train_model(model, X_train, X_test, y_train, y_test)