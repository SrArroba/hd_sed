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

###################################### METHODS ############################################
def chooseRandomFiles():
    nFiles = random.randint(2, desiredOverlap) # Number of files to merge
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

def countOverlap(annot_DF):
    # Study overlapping (more than 1 non zero value in a column)
    overlap = False 
    maxOverlap = 1
    timesOverlap = [] # Number of overlapping   
    colIndex = 0
    for column in annot_DF:
        events = []
        for val in annot_DF[column]:
            if val != 0: events.append(val) 

        # General overlap
        if len(events) > 1: 
            overlap = True
            timesOverlap.append(colIndex*0.01)
        # Max overlap
        if (len(events) > maxOverlap): 
            maxOverlap = len(events)
            print("Max at: ", column)

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

def getFileID(fileName):
    fileSplit = fileName[:-4].split("/") # Remove .csv and split in subfolders
    fileID = fileSplit[len(fileSplit)-1] # Remove the rest, keep only numbers(ID)
    return fileID

def getInputMatrix(csvDF):
    # Open species file
    speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"
    speciesDF = pd.read_csv(speciesFile) # Make it DataFrame
    speciesDF = speciesDF.iloc[1: , :] # Remove empty first row

    # Duration properties
    stdDuration = 5
    stamp = 0.01

    ###################### INITIAL MATRIX (nxm dimension) ######################
    # n = number of species
    rowNames = speciesDF["class name"].to_numpy()
    # m = time stamps
    columnValues = np.arange(0, stdDuration+stamp, stamp)
    columnNames = []
    for val in columnValues: # Convert to string and force 2 decimals
        columnNames.append(format(val, '.2f'))

    # Create matrix with 0s
    zerosMatrix = np.zeros((len(rowNames), len(columnNames)))

    # Create DataFrame
    inputDF = pd.DataFrame(zerosMatrix, columns=columnNames)
    inputDF.index = rowNames

    ########################## FILL THE INPUT MATRIX ###########################
    for line in range(len(csvDF)):
        starting = csvDF.at[line, "Start"]
        dur = csvDF.at[line, "Duration"]
        species = csvDF.at[line, "Label"]

        lineStart = math.floor(float(starting) * 100)/100.0
        lineEnd = math.floor(float(starting) * 100)/100 + math.floor(float(dur) * 100)/100 
        spec = species.replace("\n","")
       
        # Get time stamps where there is presence of the species
        timeValues = np.arange(lineStart, lineEnd+stamp, stamp)
        
        # Fill input matrix with values (1 when presence)
        for val in timeValues:
            formatVal = format(val, '.2f')
            inputDF.at[spec, formatVal] = 1

    return inputDF

def getSpectrogram(audioSegment):
    # Get samples and transform to np array
    samples = audioSegment.get_array_of_samples()
    arr = np.array(samples).astype(np.float32)
    
    # Obtain librosa mel spectrogram
    sr = 44100
    S = librosa.feature.melspectrogram(y=arr, sr=sr)

    return S

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


###################################### MAIN ###############################################

### MAIN PARAMETERS ###
stamp = 0.01
desiredOverlap = 7
stdDuration = 5

audioFolder = "../data/nips4b/audio/original/train/"
annotFolder = "../data/nips4b/annotations/"
listAnnot = os.listdir(annotFolder) # Only file name, no path
speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"

#######################

###### CHOOSE WHICH FILES ARE GOING TO BE MERGED #####    
chosenIDs = chooseRandomFiles()

##### MERGE CSV FILES #####
posList = np.zeros(len(chosenIDs))
chosenCSV = createAnnotList(chosenIDs)
csvDF = concatSeveralCSV(chosenCSV, posList)

bigDF = getInputMatrix(csvDF) # Get input dataframe

finalOv = countOverlap(bigDF)

##### CHECK IF MATCHING DESIRED OVERLAPPING #####
if(finalOv > desiredOverlap): # Take only files with less or equal polyphony as desired
    print("Over desired overlapping (",finalOv,")")
else:
    print("FINAL OVERLAP: ", finalOv)

    # Print heat map (annotations)
    plt.figure()
    sns.heatmap(bigDF)
    plt.show()

    # Merge audios and obtain spectrogram
    audioList = createAudioList(chosenIDs)
    mergedAudio = mergeAudios(audioList, posList)
    spec = getSpectrogram(mergedAudio)
    plotSpec(spec)
