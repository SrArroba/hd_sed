from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask
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


###################################### METHODS ############################################
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
            df.at[j, "Start"] += pos
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

def from_InputDF_to_SELDF(inputDF):
    columnNames = ["Start", "End", "Label"]
    selDF = df = pd.DataFrame(columns = columnNames)

    for index, row in inputDF.iterrows():
        checkZero = False
        for cindex, col in row.iteritems():
            if col == 1.0 and not checkZero: # First 1
                start = cindex
                label = index
                checkZero = True
            if col == 0.0 and checkZero: # A 0 after a 1 
                end = float(cindex)-0.01
                entry = {'Start': start, 'End': end, 'Label': label}
                selDF = selDF.append(entry, ignore_index = True)
                checkZero = False
            if cindex == "5.00" and col == 1.0: # If it ends at the last time stamp
                end = float(cindex)
                entry = {'Start': start, 'End': end, 'Label': label}
                print(entry)
                selDF = selDF.append(entry, ignore_index = True)
    
    return selDF                

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

    while(len(inFeat) != n_files): # Iterate until reaching desired amount of files
        # Choose files 
        chosenIDs = chooseRandomFiles(polyphony) # 334

        ##### MERGE CSV FILES #####
        posList = np.zeros(len(chosenIDs))
        chosenCSV = createAnnotList(chosenIDs)
        csvDF = concatSeveralCSV(chosenCSV, posList)

        inputMatrix = getInputMatrix(csvDF) # Get input dataframe

        #print("Input annotations shape: ", inputDF.shape)

        finalOv = countOverlap(inputMatrix)

        ##### CHECK IF MATCHING DESIRED OVERLAPPING #####
        if(finalOv <= polyphony): # Take only files with less or equal polyphony as desired

            # Merge audios and obtain spectrogram
            audioList = createAudioList(chosenIDs)
            mergedAudio = mergeAudios(audioList, posList)
            spec = getSpectrogram(mergedAudio)
            ## Print heat map (annotations)
            # plt.figure()
            # sns.heatmap(inputMatrix)
            # plt.show()
            #plotSpec(spec)
            # plotSpec2(spec)
            #eb = getEnergyBands(spec, inputDF, 40, 0.5)

            # Fill returning lists (input features and annotations)
            inAnnot.append(inputMatrix)
            inFeat.append(spec)
           
    inFeat = np.array(inFeat)
    #inAnnot = np.array(inAnnot)

    return inFeat, inAnnot


def getCombinedName(filesList, extension):
    finalName = ""
    for f in filesList:
        finalName = finalName + getFileID(f) + "_"
    finalName = finalName[:-1] + extension # Renove last _ and add extension

    return finalName

def getCSV_DF(csvFile):
    columnNames = ['Start','Duration', 'Label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df

def getEnergyBands(spec, annotDF, window, overlap): 
    # ----------------------------------------------
    # Window in miliseconds
    # Overlap in percetange: [0,1)
    # ----------------------------------------------

    bands = []
    step = int(window*overlap/10)
    nBands = int(len(spec[0])/step)

    for i in range(nBands):
        st = i*step
        end = st + step
       # print("STARTS; END --> ", st, " --> ", end)
        band = spec[:, st:end]
        bands.append(band)
    
    # print("Number of energy bands: ", len(bands))
    # print("Energy band shape: ", bands[0].shape)
    return bands

def getFileID(fileName):
    fileSplit = fileName[:-4].split("/") # Remove .csv and split in subfolders
    fileID = fileSplit[len(fileSplit)-1] # Remove the rest, keep only numbers(ID)
    return fileID

def getInputMatrix(csvDF):
    ###########################################################
    # Gets DF from original csv (Start, Duration, Label) format
    # and transform it to binary matrix that will be fed to the
    # model as output/annotations 
    # Output format (n_classes, time_stamps)
    ###########################################################

    # Determines how many divisions are in the time scale 
    specFeats = 430
    stamp = stdDuration/specFeats

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
    if(mat.shape[1] == (specFeats+1)):
        mat = np.delete(mat, -1, axis=1) # Remove last to match spectrogram shape

    # Check if is smaller than the standard length:
    if(mat.shape[1] < specFeats):
        colMissing = specFeats - mat.shape[1]
        matZeros = np.zeros((len(labels), colMissing))
        
        # Concatenate original matrix with the submatrix of 0s
        mat = np.concatenate((mat, matZeros), axis=1)
        
    # Print info (can be commented)
    # print(mat, " (", mat.shape, ")", type(mat))
 
    return mat

def getSpectrogram(audioSegment):
    # Get samples and transform to np array
    samples = audioSegment.get_array_of_samples()
    arr = np.array(samples).astype(np.float32)

    # Obtain librosa mel spectrogram
    S = librosa.feature.melspectrogram(y=arr, sr=sr,  power=1, win_length=win_len, hop_length=hop_len, n_mels=n_mels)
    S = np.delete(S , -1, axis=1)
    # Plot spectrogram (can be commmented)
    # plotSpec(S)    
    
    return S

def getLabelList():
    speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"
    speciesDF = pd.read_csv(speciesFile) # Make it DataFrame
    labels = speciesDF["class name"].to_numpy()
    labels = np.delete(labels, 0) # Remove label 'Empty'

    return np.sort(labels)

def get_SEL_DF(otherDF):
    columnNames = ["event_onset", "event_offset", "event_label"]
    altDF = pd.DataFrame(columns = columnNames)

    for index, row in otherDF.iterrows():
        altDF.at[index, "event_onset"] = otherDF.at[index, "Start"]
        altDF.at[index, "event_offset"] = row.loc["Start"] + row.loc["Duration"]
        altDF.at[index, "event_label"] = otherDF.at[index, "Label"]

    return altDF 

def mergeAudios(audioList, positionList): # Merge from audio list and generate an AudioSegment object
    # Generate empty audio file
    finalAudio = AudioSegment.silent(duration=1000*stdDuration)

    # Merge every recording in the list into final audio
    for i in range(len(audioList)):
        soundSeg = AudioSegment.from_file(audioList[i], format="wav")
        finalAudio = finalAudio.overlay(soundSeg, position=positionList[i])

    return finalAudio 

def plotSpec(S):
    sr = 44100
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

def plotSpec2(S):
    sr = 44100
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S[0], ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
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
win_len = 2048
hop_len = 512
n_mels = 128
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


# generateDataset(10, 3)