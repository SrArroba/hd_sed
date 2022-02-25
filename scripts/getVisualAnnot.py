import os 
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Paths 
annotationsFolder = os.path.expanduser("../data/nips4b/annotations/")
audioFolder = os.path.expanduser("../data/nips4b/audio/original/train/")

testFile = os.path.join(annotationsFolder, "annotation_train004.csv")

# Data frame building
columnNames = ['Start','Duration', 'Label']
df = pd.read_csv(testFile, names = columnNames)

print(df)

megaList = []
# Check every entry in annotation file
for i in range(len(df)):
    listTime = np.arange(round(df['Start'][i], 2), df['Start'][i]+df['Duration'][0], 0.01).tolist()
    listTime = [round(i, 2) for i in listTime]
    listTime = np.insert(str(listTime), 0, df['Label'][i])
    megaList.append(listTime)

###################################################
testAudio = os.path.join(audioFolder, "nips4b_birds_trainfile004.wav")
# Check duration of file 
duration = librosa.get_duration(filename=testAudio)
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
detectionMatrix = [] #np.empty(len(allLabels), dtype=np.ndarray)
for label in allLabels:
    # Empty temporal row 
    emptyRow = np.zeros(np.shape(timeStamps))
    
    for entry in megaList:
        if entry[0] == label:
            entry[1] = entry[1].replace("[", "")
            entry[1] = entry[1].replace("]", "")
            tVals = entry[1].split(",")

            for time in tVals:
                timeIndex = int(float(time)/0.01)
                emptyRow[timeIndex] = int(allLabels.index(label)+1)
    
    #print(emptyRow)
    detectionMatrix.append(emptyRow)

print(detectionMatrix)

###########################################  PLOTTING  ###########################################
import seaborn as sns

timeStr = [str(x) for x in timeStamps]
pltHeaders = timeStr
print(len(pltHeaders))
print(len(detectionMatrix[0]))
print(len(detectionMatrix[1]))
print(len(detectionMatrix[2]))

df = pd.DataFrame(detectionMatrix, columns=pltHeaders)
df.index = allLabels
df.index.name = 'Species'
print(df)

sns.heatmap(df)
plt.show()x 