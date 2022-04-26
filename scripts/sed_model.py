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
from evaluation import f1_score_all, er_all, get_score, sed_eval_scores, plotPredTruth


########################################################################
#                            FUNCTIONS                                 # 
########################################################################
def generate_sets(n_files, polyphony):
    # Generate entire dataset
    # train_x, train_y = processing.generateDataset(n_files, polyphony)
    # print("NIPS: ", train_x.ndim, train_x[0].shape, type(train_x[0]), train_x[0].ndim, train_y[0].shape, type(train_y[0]), train_y[0].ndim)
    train_x, train_y = dryad_process.generateDataset(n_files, polyphony)

    # Separate into validation and training sets
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

    # Pre-process 
    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test, y_train, y_test):
    X_train = np.array([np.array(val, dtype='float') for val in X_train])
    y_train = np.array([np.array(val, dtype='float') for val in y_train])
    X_test = np.array([np.array(val, dtype='float') for val in X_test])
    y_test = np.array([np.array(val, dtype='float') for val in y_test])

    # for i in range(len(X_train)):
    #     print("TRAIN: ", X_train[i].shape, type(X_train[i]), " --> ", y_train[i].shape, type(y_train[i]))

    # Reshape sets: (length of set, frames, mels, 1)
    X_train = X_train.reshape(len(X_train), X_train[0].shape[0], X_train[0].shape[1], 1)
    X_test = X_test.reshape(len(X_test), X_train[0].shape[0], X_train[0].shape[1], 1)
    # Change output to categorical
    #print(y_train[0].shape)
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def create_model(in_shape, out_shape):
    # Model
    model = Sequential()
    
    # Add model layers
    model.add(Input(shape=(in_shape[-3], in_shape[-2], in_shape[-1])))

    # Convolutional part
    for i in range(len(filt_list)):
        model.add(Conv2D(filt_list[i], kernel_size=(1,1)))
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(MaxPooling2D(pool_size=(1, pool_list[i])))
        # model.add(Dropout(drop_rate))   
    
    model.add(Permute((2,1,3)))
    model.add(Reshape((in_shape[-3], -1)))

    # Recurrent part 
    for r in rnn_nodes:
        model.add(Bidirectional(GRU(r, activation='tanh', dropout=drop_rate, recurrent_dropout=drop_rate, return_sequences=True)))
    
    # Fully Connected part
    for fcn in hid_nodes:
        model.add(TimeDistributed(Dense(fcn)))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(drop_rate))
    
    # Output
    model.add(TimeDistributed(Dense(n_classes, activation='sigmoid')))

    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    #compile model
    optimizer = Adam(lr=1e-3)
    metric_list = ['accuracy']#, f1_score_all, er_all]
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metric_list)
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

########################################################################
#                                 MAIN                                 # 
########################################################################

############################## PARAMETERS ##############################

# Main values
n_classes = 200

n_files = 20
polyphony = 6

threshold = 0.5 # To create binary output matrix

#### Model params ####
epochs = 120
drop_rate = 0.5
batch_size = 128 # Batch size

# Convolutional params 
filt_list = [64, 128, 128, 128, 256]
pool_list = [5, 2, 2, 2, 2]
# Recurrent params  
rnn_nodes = [128, 128]
# FC 
hid_nodes = [128, 128]
########################################################################

#### Sets separation ####
X_train, X_test, y_train, y_test = generate_sets(n_files, polyphony)

in_shape = X_train[0].shape
out_shape = y_train[0].shape
print("\nINPUT SIZE: ", X_train[0].shape)
print("OUTPUT SIZE: ", y_train[0].shape)

###############################################

#### Create model // Load model ####
model = create_model(in_shape, out_shape)
model = load_model('dryad_day1_o6.h5')
print(model.summary())

# Train 
# train_model(model, X_train, X_test, y_train, y_test)

# Save model
# model.save('dryad_day1_o10.h5')  # HDF5 file 

###############################################

# Prediction
print("\nGenerating a prediction...")
sedeval_f1_event = []
sedeval_f1_seg = []
sedeval_er_event = []
sedeval_er_seg = []
f1s = []
ers = []
f1s_o3 = []
ers_o3 = []
f1s_o6 = []
ers_o6 = []
f1s_o10 = []
ers_o10 = []
f1s_class = []
ers_class = []
cnt = 0
for elem in X_test:
    elem = elem.reshape(1, in_shape[-3], in_shape[-2], in_shape[-1])
    prediction = model.predict(elem)

    # Prepare prediction, thresholded prediction and ground truth for evaluation
    predNoThres = prediction.reshape(out_shape[-2], out_shape[-1])    
    predNoThres = np.array(predNoThres)
    predBinary = processing.output_to_binary(prediction, threshold)
    predBinary = predBinary.reshape(out_shape[-2], out_shape[-1])

    ov = processing.countOverlap(y_test[cnt].T)

    #### Evaluation Metrics #### 
    f1, er = get_score(predBinary.T, y_test[cnt].T)
    f1s.append(f1)
    ers.append(er)

    if(ov == 3): 
        f1s_o3.append(f1)
        ers_o3.append(er)
    if(ov == 6): 
        f1s_o6.append(f1)
        ers_o6.append(er)
    if(ov == 10): 
        f1s_o10.append(f1)
        ers_o10.append(er)

    # Species based score - UNCOMMNET FOR CLASS BASED METRICS (WIP)
    # predBinClass = predBinary.T[10,:] #EATO row
    # truthClass = y_test[cnt].T[10,:]

    # f1_class, er_class = get_score(predBinClass, truthClass)

    # f1s_class.append(f1_class)
    # ers_class.append(er_class)
    # SED_EVAL library scores
    scores = sed_eval_scores(predBinary.T, y_test[cnt].T, dryad_process.speciesList)
    f1_ev = scores[0]['f_measure']['f_measure']
    er_ev = scores[0]['error_rate']['error_rate']
    f1_seg = scores[1]['f_measure']['f_measure']
    er_seg = scores[1]['error_rate']['error_rate']
    
    sedeval_f1_event.append(f1_ev)
    sedeval_er_event.append(er_ev)
    sedeval_f1_seg.append(f1_seg)
    sedeval_er_seg.append(er_seg)

    # PLOT PRED TRUTH FEATURE
    # print("File with polyphony: ", ov)
    # print("F1 and ER (manual): ", f1, er)
    # print("F1 and ER sed_eval: ", f1_seg, er_seg)
    # sns.heatmap(predNoThres.T, cbar=True)
    # plotPredTruth(predBinary.T, predNoThres.T, y_test[cnt].T, dryad_process.speciesList)
    cnt += 1

print("\nF1s: ", f1s)
print("ERs: ", ers)
###### UNCOMMENT FOR METRICS OF 3, 6 and 10 polyphony
# print("Mean F1 (3): ", np.mean(f1s_o3), np.std(f1s_o3))
# print("Mean ER (3): ", np.mean(ers_o3), np.std(ers_o3))
# print("Mean F1 (6): ", np.mean(f1s_o6), np.std(f1s_o6))
# print("Mean ER (6): ", np.mean(ers_o6), np.std(ers_o6))
# print("Mean F1 (10): ", np.mean(f1s_o10), np.std(f1s_o10))
# print("Mean ER (10): ", np.mean(ers_o10), np.std(ers_o10))
# print("Mean F1 (All): ", np.mean(f1s), np.std(f1s))
# print("Mean ER (All): ", np.mean(ers), np.std(ers))


##### UNCOMMENT FOR CLASS BASED METRICS 
#### Change class 
# print("CLASS BASED")
# print("Mean F1 class: ", np.mean(f1s_class), np.std(f1s_class))
# print("Mean ER class: ", np.mean(ers_class), np.std(ers_class))

#
# print("SED EVAL VALUES SEGMENT BASED: ")
# print("F1s: ", sedeval_f1_seg)
# print("Mean F1: ", np.mean(sedeval_f1_seg), np.std(sedeval_f1_seg))
# print("Mean ER: ", np.mean(sedeval_er_seg), np.std(sedeval_er_seg))