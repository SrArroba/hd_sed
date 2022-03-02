import os 
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import math 
import seaborn as sns
import sys

# Paths 
# Arguments:
# 0.- .py file name 
# 1.- Audio final folder
# 2.- Annotations final folder
argAudio = sys.argv[1]
argAnnot = sys.argv[2]

pathAudio = "../data/nips4b/audio/"+argAudio
pathAnnot = "../data/nips4b/"+argAnnot

annotationsFolder = os.path.expanduser(pathAnnot)
annotationList = os.listdir(annotationsFolder)
audioFolder = os.path.expanduser(pathAudio)

# Data frame building

# Iterate each annotation file
for annotationFile in annotationList:
    fileID = annotationFile[:-4]

    columnNames = ['Start','Duration', 'Label']
    annotFile = os.path.join(annotationsFolder, annotationFile)
    df = pd.read_csv(annotFile, names = columnNames)
    print(df)
    megaList = []

    # Check every entry in annotation file
    for i in range(len(df)):
        roundedStart = math.floor(df['Start'][i] * 100)/100.0
        roundedEnd = math.floor(df['Start'][i] * 100)/100 + math.floor(df['Duration'][i] * 100)/100 + 0.01

        listTime = np.arange(roundedStart, roundedEnd, 0.01).tolist()
        listTime = [round(i, 2) for i in listTime]
        listTime = np.insert(str(listTime), 0, df['Label'][i])

        megaList.append(listTime)

    ###################################################
    audioFile = fileID+".wav"
    audioFile = os.path.join(audioFolder, audioFile)
    # Check duration of file 
    duration = librosa.get_duration(filename=audioFile) +0.01
    print(duration)
    # Take only 2 decimals 
    duration = round(duration, 2)

    # Check species in the recording
    allLabels = []
    for label in df['Label']:
        if label not in allLabels:
            allLabels.append(label)

    # Get timestamps and create empty(0) 'binary' matrix
    timeStamps = np.arange(0, duration+1, 0.01).tolist()
    timeStamps = [round(i, 2) for i in timeStamps]
    detectionMatrix = np.zeros(shape= (len(allLabels), len(timeStamps)))

    for label in allLabels:
        # Empty temporal row (all 0's)
        emptyRow = np.zeros(np.shape(timeStamps))
        
        # Check in the initial DF the entries of the same label (for each loop)
        for entry in megaList:
            if entry[0] == label:
                entry[1] = entry[1].replace("[", "")
                entry[1] = entry[1].replace("]", "")
                tVals = entry[1].split(",")

                # Obtain values (1, 2, 3,...) of the species and put them in their corresponding row
                for time in tVals:
                    timeIndex = int(round(100*float(time)))-1

                    emptyRow[timeIndex] = allLabels.index(label)+1
        
            detectionMatrix[allLabels.index(label)] = emptyRow

    ###########################################  PLOTTING  ###########################################

    timeStr = [str(x) for x in timeStamps]
    pltHeaders = timeStr

    dfFinal = pd.DataFrame(detectionMatrix, columns=pltHeaders)
    dfFinal.index = allLabels
    dfFinal.index.name = 'Species'


    print("Creating detection plot for file {:s}... ".format(fileID))
    plt.figure()
    sns.heatmap(dfFinal)
    plt.show()
    # fName = '../data/nips4b/visualAnnotations/'+"visualAnnotation_"+fileID+".png"
    # plt.savefig(fName)