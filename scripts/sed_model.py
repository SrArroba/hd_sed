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
from tensorflow.keras.models import Model, Sequential, load_model 
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Flatten, BatchNormalization, Permute
from tensorflow.keras.optimizers import schedules, Adam

# Local imports 
import processing
import evaluation

########################################################################
#                            FUNCTIONS                                 # 
########################################################################
def generate_sets(n_files, polyphony):
    # Generate entire dataset
    train_x, train_y = processing.generateDataset(n_files, polyphony)

    # Separate into validation and training sets
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

    # Pre-process 
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test, y_train, y_test):
    
    # Reshape sets: (length of set, frames, mels, 1)
    X_train = X_train.reshape(len(X_train), time_stamps, n_mels, 1)
    X_test = X_test.reshape(len(X_test),time_stamps, n_mels, 1)
    # Change output to categorical
    #print(y_train[0].shape)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def create_model(in_shape, out_shape):
    model = Sequential()
    #add model layers
    # model.add(Input(shape=(time_stamps, n_mels, 1)))
    model.add(Input(shape=(in_shape[-3], in_shape[-2], in_shape[-1])))

    # Convolutional part
    for i in range(len(filt_list)):
        model.add(Conv2D(filt_list[i], kernel_size=(1,1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, pool_list[i])))
        model.add(Dropout(drop_rate))   
    
    model.add(Permute((2,1,3)))
    model.add(Reshape((in_shape[-3], -1)))

    # Recurrent part 
    for r in rnn_nodes:
        model.add(Bidirectional(GRU(r, activation='tanh', dropout=drop_rate, recurrent_dropout=drop_rate, return_sequences=True)))
    
    # Fully Connected part
    for fcn in hid_nodes:
        model.add(TimeDistributed(Dense(fcn)))
        model.add(Dropout(drop_rate))
    
    # Output
    model.add(TimeDistributed(Dense(n_classes, activation='sigmoid')))
   
    print(model.summary())

    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    #compile model
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)#, validation_data=(X_test, y_test))

########################################################################
#                                 MAIN                                 # 
########################################################################

############################## PARAMETERS ##############################

# Main values
n_classes = 87
time_stamps = 216
n_mels = 128

n_files = 10000
polyphony = 3 

threshold = 0.5 # To create binary output matrix

#### Model params ####
epochs = 100
drop_rate = 0.5
batch_size = 64 # Batch size

# Convolutional params 
filt_list = [128, 128, 128]
pool_list = [5, 2, 2]
# Recurrent params 
rnn_nodes = [32, 32]
# FC 
hid_nodes = [32]
########################################################################

#### Sets separation ####
X_train, X_test, y_train, y_test = generate_sets(n_files, polyphony)

in_shape = X_train[0].shape
out_shape = y_train[0].shape
print("INPUT SIZE: ", X_train[0].shape)
print("OUTPUT SIZE: ", y_train[0].shape)

#### Create model // Load model ####
model = create_model(in_shape, out_shape)
# model = load_model('baseline_night_25_03.h5')

# Train 
train_model(model, X_train, X_test, y_train, y_test)

# Save model
model.save('model_spec_02_decay.h5')  # HDF5 file 
# Prediction
print("Generate a prediction")
for elem in X_test:
    elem = elem.reshape(1, in_shape[-3], in_shape[-2], in_shape[-1])
    prediction = model.predict(elem)
    print("Max value no Binary: ", np.amax(prediction))
    predBinary = processing.output_to_binary(prediction, threshold)

    # print("PREDICTION: ", predBinary)
    # print("Max value: ", np.amax(predBinary))

    #### Evaluation Metrics #### 
    f1, er = evaluation.get_score(predBinary, y_test[0])

    print("SCORES: ")
    print("F1 Score     --> ", f1)
    print("Error Rate   --> ", er)