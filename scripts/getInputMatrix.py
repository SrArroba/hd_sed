import os 
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def getCSV_DF(csvFile):
    columnNames = ['Start','Duration', 'Label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df


#############################################################################
csvFile = "../data/nips4b/annotations/612.csv"
#############################################################################

# Open species file
speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"
speciesDF = pd.read_csv(speciesFile) # Make it DataFrame
speciesDF = speciesDF.iloc[1: , :] # Remove empty first row

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
inputDF.to_csv('./inputMATRIX.csv')

plt.figure()
sns.heatmap(inputDF)
plt.show()

