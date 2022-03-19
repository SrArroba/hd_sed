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
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Flatten, BatchNormalization, Permute

# Local imports 
import processing

########################################################################
#                            FUNCTIONS                                 # 
########################################################################
def generate_sets():
    # Generate entire dataset
    n_files = 1000
    polyphony = 3 
    train_x, train_y = processing.generateDataset(n_files, polyphony)

    # Separate into validation and training sets
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

    # Pre-process 
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test, y_train, y_test):
    print(X_train.shape)
    print(X_test.shape)
    # Reshape sets
    X_train = X_train.reshape(len(X_train), n_mels)
    X_test = X_test.reshape(len(X_test), n_mels)
    # Change output to categorical
    #print(y_train[0].shape)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def create_model(in_shape, out_shape):
    model = Sequential()
    #add model layers
    model.add(Input(shape=(n_mels, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Convolutional part
    # model.add(Conv2D(cnn_filters, kernel_size=(1,1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1,5)))
    # model.add(Dropout(drop_rate))
    # model.add(Conv2D(cnn_filters, kernel_size=(1,1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1,2)))
    # model.add(Dropout(drop_rate))
    # model.add(Conv2D(cnn_filters, kernel_size=(1,1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1,2)))
    # model.add(Dropout(drop_rate))
   
    # model.add(Permute((2,1,3)))
    # model.add(Reshape((in_shape[-2], -1)))
    # model.add(Dense(n_classes, activation='softmax'))
    #model.add(Activation('softmax', name='strong_out'))
   
    print(model.summary())

    #compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
########################################################################
#                                 MAIN                                 # 
########################################################################

# Main values
n_classes = 87
time_stamps = 216
n_mels = 128

# Model params
drop_rate = 0.5
# Convolutional params 
cnn_filters = 128


X_train, X_test, y_train, y_test = generate_sets()
in_shape = X_train[0].shape
print("INPUT SIZE: ", X_train[0].shape)
print("OUTPUT SIZE: ", y_train[0].shape)
model = create_model(in_shape, 1)

train_model(model, X_train, X_test, y_train, y_test)