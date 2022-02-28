import os 
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import math 
import seaborn as sns

# Paths 
annotationsFolder = os.path.expanduser("../data/nips4b/annotations/")
annotationList = os.listdir(annotationsFolder)
audioFolder = os.path.expanduser("../data/nips4b/audio/original/train/")

# Data frame building
columnNames = ['Start','Duration', 'Label']

# Iterate each annotation file
for annotationFile in annotationList:
    fileID = annotationFile.replace("annotation_train", "")
    fileID = fileID.replace(".csv", "")

    annotFile = os.path.join(annotationsFolder, annotationFile)
    df = pd.read_csv(annotFile, names = columnNames)

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
    audioFile = "nips4b_birds_trainfile"+fileID+".wav"
    audioFile = os.path.join(audioFolder, audioFile)
    # Check duration of file 
    duration = librosa.get_duration(filename=audioFile) +0.01
    # Take only 2 decimals 
    duration = round(duration, 2)

    # Check species in the recording
    allLabels = []
    for label in df['Label']:
        if label not in allLabels:
            allLabels.append(label)

    # Modify matrix to values in classes
    timeStamps = np.arange(0, duration, 0.01).tolist()
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
                    timeIndex = int(float(time)/0.01) - 1
                    
                    emptyRow[timeIndex] = int(allLabels.index(label)+1)
        
            detectionMatrix[allLabels.index(label)] = emptyRow

###########################################  GET OVERLAPPING  ###########################################

    # Column headers
    timeStr = [str(x) for x in timeStamps]
    pltHeaders = timeStr

    # Create dataframe
    dfFinal = pd.DataFrame(detectionMatrix, columns=pltHeaders)
    dfFinal.index = allLabels
    dfFinal.index.name = 'Species'

    # Study overlapping (more than 1 non zero value in a column)
    overlap = False 
    maxOverlap = 1
    timesOverlap = [] # Number of overlapping   
    colIndex = 0
    for column in dfFinal:
        events = []
        for val in dfFinal[column]:
            if val != 0: events.append(val) 

        # General overlap
        if len(events) > 1: 
            overlap = True
            timesOverlap.append(colIndex*0.01)
        # Max overlap
        if (len(events) > maxOverlap): maxOverlap = len(events)

        colIndex += 1
   
    print("\nOverlapping study of audio file {:d}: ".format(int(fileID)))
    print("Appearing species: ", len(allLabels), "(", allLabels,")")
    print("Is there overlapping? ", overlap)
    print("Maximum overlapping: ", maxOverlap)
    print("Overlapping percentage: ", 100*len(timesOverlap)/len(timeStamps))