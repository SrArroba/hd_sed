from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from pydub import AudioSegment
import random
import seaborn as sns
import sed_eval
from random import randint
from sklearn import preprocessing
import scipy.signal

###################################### METHODS ############################################
def build_mbe(audio_data, sr, nfft, n_mels):
    ###############################
    # Output: (n_mels, n_frames)
    ###############################
    y = audio_data

    spec, n_fft = librosa.core.spectrum._spectrogram(y=y, n_fft=nfft, hop_length=nfft//2, power=1)

    mel = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=n_mels)

    # Dot product of mel and spec
    dot_prod = np.dot(mel, spec)
    dot_prod[dot_prod < 1] = 1
    
    # Tranform to log scale
    log_out = np.log(dot_prod)
    
    return log_out

def chooseRandomFiles(polyphony):
    #### Return a list of file IDs with length between 2 and the desired overlap 

    nFiles = random.randint(2, polyphony) # Number of files to merge
    chosenIDs = [] # List of chosen IDS to merge

    for i in range(nFiles):
        rdmFile = random.choice(listAnnot) # Choose random file from annotation folder
        fileID = getFileID(rdmFile)
        
        if(fileID in chosenIDs): # Check that the file is not already considered
            continue
        
        chosenIDs.append(fileID)
    
    return chosenIDs

def concatSeveralCSV(csvList, positionList):
    # Create a list of DataFrames 
    frames = []

    for i in range(len(csvList)):
        df = getCSV_DF(csvList[i]) # Get DF
        pos = positionList[i]
        
        # Change start value (position value)
        for j in range(len(df)):
            df.at[j, "event_onset"] += pos
        frames.append(df)

    # Concatenate list of DF into a single DF
    result = pd.concat(frames, ignore_index=True) # Ignore index = continues index list without starting from 0 again

    return result

def countOverlap(annot_Matrix):
    # Study overlapping (more than 1 non zero value in a column)
    overlap = False 
    maxOverlap = 1
    timesOverlap = [] # Number of overlapping   
    colIndex = 0
    for column in annot_Matrix.T:
        events = []
        for val in column:
            if val != 0: events.append(val) 

        # General overlap
        if len(events) > 1: 
            overlap = True
            timesOverlap.append(colIndex*0.01)
        # Max overlap
        if (len(events) > maxOverlap): 
            maxOverlap = len(events)

        colIndex += 1

    return maxOverlap

def createAnnotList(listID):
    annotList = []
    for i in listID:
        annotList.append(annotFolder+i+".csv")
    
    return annotList

def createAudioList(listID):
    audioList = []
    for i in listID:
        audioList.append(audioFolder+i+".wav")
    
    return audioList

def features_to_windows(feature):
    windows = []
    for win in feature.T:
        windows.append(win)

    return windows            

def generateDataset(n_files, polyphony):
    # -----------------------------------------------------------
    # n_files : Desired number of files that compose the dataset 
    # polyphony: Maximum polyphony that the dataset has 
    # 
    # Returns a list of spectrograms and a list of annotations 
    # with length = n_files 
    # -----------------------------------------------------------

    inFeat = []
    inAnnot = []
    count = 0
    while(count != n_files): # Iterate until reaching desired amount of files
        # Choose files 
        chosenIDs = chooseRandomFiles(polyphony)
        posList = np.zeros(len(chosenIDs))
        
        ##### MERGE AUDIOS and OBTAIN FEATURE #####
        audioList = createAudioList(chosenIDs)
        mergedAudio = mergeAudios(audioList, posList)
        
        # feat = getSpectrogram(mergedAudio)
        feat = getMelSpectrogram(mergedAudio)
        # feat = build_mbe(mergedAudio, sr, win_len, n_mels)

        ##### MERGE CSV FILES #####
        chosenCSV = createAnnotList(chosenIDs)
        csvDF = concatSeveralCSV(chosenCSV, posList)

        inputMatrix = getInputMatrix(csvDF, feat.shape[1]) # Get input dataframe

        # print("Input annotations shape: ", inputMatrix.shape)

        finalOv = countOverlap(inputMatrix)

        ##### CHECK IF MATCHING DESIRED OVERLAPPING #####
        if(finalOv <= polyphony): # Take only files with less or equal polyphony as desired
            count += 1
            print("Generating dataset: {} out of {} ({} %)".format(count, n_files, (100*count/n_files)), end='\r')
            
            ## PRINT heat map (annotations) and spectrogram
            # plt.figure()
            # sns.heatmap(inputMatrix)
            # plt.show()
            # plotSpec(feat)

            # Normalize features
            # print("MIN-MAX NO NORM: ", np.amin(feat), np.amax(feat))
            # feat = (feat - feat.mean()) / feat.var()

            feat = normalize_data(feat.T)
            # print("MIN-MAX NORM: ", np.amin(feat), np.amax(feat))
            # Fill returning lists (input features and annotations)
            
            inFeat.append(feat)
            inAnnot.append(inputMatrix.T)

    inFeat = np.array(inFeat)
    inAnnot = np.array(inAnnot)

    print("Input shape: ", inFeat[0].shape, "(", type(inFeat[0]), ")")
    print("Annot shape: ", inAnnot[0].shape, "(", type(inAnnot[0]), ")")
    
    return inFeat, inAnnot


def getCombinedName(filesList, extension):
    finalName = ""
    for f in filesList:
        finalName = finalName + getFileID(f) + "_"
    finalName = finalName[:-1] + extension # Renove last _ and add extension

    return finalName

def getCSV_DF(csvFile):
    columnNames = ['event_onset','event_duration', 'event_label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df

def getFileID(fileName):
    fileSplit = fileName[:-4].split("/") # Remove .csv and split in subfolders
    fileID = fileSplit[len(fileSplit)-1] # Remove the rest, keep only numbers(ID)
    return fileID

def getInputMatrix(csvDF, n_seps):
    ###########################################################
    # Gets DF from original csv (Start, Duration, Label) format
    # and the time seàraton and transform it to binary matrix 
    # that will be fed to the model as output/annotations 
    # Output format (n_classes, time_stamps)
    ###########################################################

    # Determines how many divisions are in the time scale 
    stamp = stdDuration/n_seps

    # Transform to [onset, offset, label] (SEL) dataframe
    df_sel = get_SEL_DF(csvDF)
    
    # Tranform to a records dictionary
    # E.g. {onset: "", offset: "", label: ""}
    dict1 = df_sel.to_dict('records')   

    # Get Labels (bird species)
    labels = getLabelList().tolist()

    # Obtain binary matrix 
    mat = sed_eval.util.event_roll.event_list_to_event_roll(dict1, labels, stamp)
    mat = np.transpose(mat) # Transpose
    if(mat.shape[1] == (n_seps+1)):
        mat = np.delete(mat, -1, axis=1) # Remove last to match spectrogram shape

    # Check if is smaller than the standard length:
    if(mat.shape[1] < n_seps):
        colMissing = n_seps - mat.shape[1]
        matZeros = np.zeros((len(labels), colMissing))
        
        # Concatenate original matrix with the submatrix of 0s
        mat = np.concatenate((mat, matZeros), axis=1)
        
    # Print info (can be commented)
    # print(mat, " (", mat.shape, ")", type(mat))
 
    return mat

def getMelSpectrogram(audioSegment):
    # Get samples and transform to np array
    #samples = audioSegment.get_array_of_samples()
    arr = np.array(audioSegment).astype(np.float32)

    # Obtain librosa mel spectrogram
    S = librosa.feature.melspectrogram(y=arr, sr=sr,  power=1, win_length=win_len, hop_length=hop_len, n_mels=n_mels)

    # Power to decibels
    S_dB = librosa.amplitude_to_db(S, ref=np.max, top_db=85, amin=1e-05)

    #S = np.delete(S , -1, axis=1)
    # Plot spectrogram (can be commmented)
    # plotSpec(S_dB)    
    
    return S_dB

def getLabelList():
    speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"
    speciesDF = pd.read_csv(speciesFile) # Make it DataFrame
    labels = speciesDF["class name"].to_numpy()
    labels = np.delete(labels, 0) # Remove label 'Empty'

    return np.sort(labels)

def getSpectrogram(audioSegment):
    # Trasform audio segment samples to numpy array
    arr = np.array(audioSegment).astype(np.float32)
    # ShortTimeFourierTransform
    _, _, spec = scipy.signal.stft(x=arr,
                                    fs=sr,
                                    nperseg=win_len,
                                    noverlap=hop_len,
                                    nfft=win_len)
    # Amplitudes
    amps = np.abs(spec)
    # Spectrogram
    spec = librosa.amplitude_to_db(S=amps,
                                   ref=np.max,
                                   amin=1e-05,
                                   top_db=85)

    return spec


def get_SEL_DF(otherDF):
    columnNames = ["event_onset", "event_offset", "event_label"]
    altDF = pd.DataFrame(columns = columnNames)

    for index, row in otherDF.iterrows():
        altDF.at[index, "event_onset"] = otherDF.at[index, "event_onset"]
        altDF.at[index, "event_offset"] = row.loc["event_onset"] + row.loc["event_duration"]
        altDF.at[index, "event_label"] = otherDF.at[index, "event_label"]

    return altDF 

def mergeAudios(audioList, positionList): # Merge from audio list and generate an AudioSegment object
    # Generate empty audio file
    finalAudio = AudioSegment.silent(duration=1000*stdDuration)

    # Merge every recording in the list into final audio
    for i in range(len(audioList)):
        soundSeg = AudioSegment.from_file(audioList[i], format="wav")
        finalAudio = finalAudio.overlay(soundSeg, position=positionList[i])
    
    samples = finalAudio.get_array_of_samples()
    finalAudio = np.array(samples).astype(np.float32)
    
    # Data augmentation
    augment = Compose([
        #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.15, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    finalAudio = augment(samples=finalAudio, sample_rate=sr)

    return finalAudio 

def normalize_data(feature):
    stdScal = preprocessing.StandardScaler()
    feature = stdScal.fit_transform(feature)
    #feature = preprocessing.normalize(feature, norm='l2')
    # mean = np.mean(feature)
    # std = np.std(feature)

    # for elem in feature: 
    #     elem = (elem-mean)/std
    
    # print("MEAN Y STD: ", np.mean(feature), np.std(feature))
    return feature

def norm_feat(feat):
    return (val-mean)/std

def output_to_binary(outMatrix, threshold):
    outMatrix[outMatrix > threshold] = 1
    outMatrix[outMatrix < threshold] = 0
    
    return outMatrix

def plotPredTruth(pred, truth):
    # Transform to DF
    pred = evaluation.from_annotMatrix_to_annotDF(pred)
    truth = evaluation.from_annotMatrix_to_annotDF(truth)

    # Plot
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.subplots_adjust(wspace=0.01)

    sns.heatmap(pred, cmap="hot", ax=ax1, cbar=False)
    sns.heatmap(truth, cmap="hot", ax=ax2, cbar=False)

    ax2.yaxis.tick_right()

    fig.subplots_adjust(wspace=0.001)
    plt.show()
  
def plotSpec(S):
    sr = 22050
    fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


###################################### MAIN ###############################################

### MAIN PARAMETERS ###
stdDuration = 5

audioFolder = "../data/nips4b/audio/original/train/"
annotFolder = "../data/nips4b/annotations/"
listAnnot = os.listdir(annotFolder) # Only file name, no path
speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"

# Spectrogram values
sr = 22050
win_len = 1024
hop_len = 512
n_mels = 40
fr = sr/hop_len
# print("###################################")
# print("             AUDIO DATA: ")
# print("Sample rate: ", sr)
# print("Frame rate: ", fr)
# print("Window lenght: ", win_len)
# print("Hop length: ", hop_len)
# print("Mels: ", n_mels)
# print("###################################")


#######################


# inFeat, inAnnot = generateDataset(1, 3)