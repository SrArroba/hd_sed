import os
import librosa
from pydub import AudioSegment
import random
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

###################################### METHODS ############################################
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
        if (len(events) > maxOverlap): maxOverlap = len(events)

        colIndex += 1

    return maxOverlap

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
    print(inputDF)

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

def getFileID(fileName):
    fileSplit = fileName[:-4].split("/") # Remove .csv and split in subfolders
    fileID = fileSplit[len(fileSplit)-1] # Remove the rest, keep only numbers(ID)
    return fileID

def getCombinedName(filesList, extension):
    finalName = ""
    for f in filesList:
        finalName = finalName + getFileID(f) + "_"
    finalName = finalName[:-1] + extension # Renove last _ and add extension

    return finalName

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
    result = pd.concat(frames, ignore_index=True)
    print(result)

    return result

def mergeAudios(audioList, positionList):
    # Get final audio file name
    audioName = getCombinedName(audioList, ".wav")

    # Generate empty audio file
    finalAudio = AudioSegment.silent(duration=1000*stdDuration)

    # Merge every recording in the list into final audio
    for i in range(len(audioList)):
        soundSeg = AudioSegment.from_file(audioList[i], format="wav")
        finalAudio = finalAudio.overlay(soundSeg, position=positionList[i])
    
    # Export audio file (i.e. save it)
    finalPath = outAudioFolder+audioName
    file_handle = finalAudio.export(finalPath, format="wav")

    return audioName 

def getCSV_DF(csvFile):
    columnNames = ['Start','Duration', 'Label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df


###################################### MAIN ###############################################

### MAIN PARAMETERS ###
stamp = 0.01
desiredOverlap = 3
stdDuration = 5

audioFolder = "../data/nips4b/audio/original/train/"
annotFolder = "../data/nips4b/annotations/"
listAnnot = os.listdir(annotFolder) # Only file name, no path
speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"

outAudioFolder = "../data/nips4b/audio/o3/"
outAnnotFolder = "../data/nips4b/mergedAnnotations/"

#######################

chosenIDs = []
combPol = 0
rdmFile = "612.csv" #random.choice(listAnnot)
chosenIDs.append(annotFolder+rdmFile)
fileDF = getCSV_DF(annotFolder+rdmFile)
inputDF = getInputMatrix(fileDF)

polyphony = countOverlap(inputDF)
combPol += polyphony

###### CHOOSE WHICH FILES ARE GOING TO BE MERGED #####
if(polyphony == desiredOverlap):
    print("File already generated with polyphony of ",desiredOverlap, " (",rdmFile,")")

else:
    polReached = False
    while(not polReached): # Repeat until reaching desired 
        rdmFile = "010.csv" #random.choice(listAnnot)
        
        if (rdmFile in chosenIDs): # Check that the file is not already analyzed
            continue
        fileDF = getCSV_DF(annotFolder+rdmFile)
        newDF = getInputMatrix(fileDF)
        newOv = countOverlap(newDF)
        
        if(newOv + combPol > desiredOverlap): # Check if it doesn't get a polyphony higher than desired
            continue
        
        chosenIDs.append(annotFolder+rdmFile)
        combPol += newOv

        if(combPol == desiredOverlap): # When reaching the desired polyphony
            polReached = True
        
    print(chosenIDs)
    
##### MERGE THOSE FILES #####
posList = np.zeros(len(chosenIDs))
audioList = []
for ann in chosenIDs:
    fileID = getFileID(ann)
    audioList.append(audioFolder+fileID+".wav")

csvDF = concatSeveralCSV(chosenIDs, posList)
audioPath = mergeAudios(audioList, posList)

##### GET INPUT DATAFRAME #####
inputDF = getInputMatrix(csvDF)

#print(inputDF)

plt.figure()
sns.heatmap(inputDF)
plt.show()