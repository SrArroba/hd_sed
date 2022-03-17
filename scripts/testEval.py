import os 
import pandas as pd 
import numpy as np 
import math
import sed_eval
import dcase_util

def getCSV_DF(csvFile):
    columnNames = ['Start','Duration', 'Label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df

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

def get_SEL_DF(otherDF):
    columnNames = ["Start", "End", "Label"]
    altDF = df = pd.DataFrame(columns = columnNames)

    for index, row in otherDF.iterrows():
        altDF.at[index, "Start"] = otherDF.at[index, "Start"]
        altDF.at[index, "End"] = row.loc["Start"] + row.loc["Duration"]
        altDF.at[index, "Label"] = otherDF.at[index, "Label"]

    return altDF

def from_InputDF_to_SELDF(inputDF):
    columnNames = ["event_onset", "event_offset", "event_label"]
    selDF = df = pd.DataFrame(columns = columnNames)

    for index, row in inputDF.iterrows():
        checkZero = False
        for cindex, col in row.iteritems():
            if col == 1.0 and not checkZero: # First 1
                start = math.floor(float(cindex)*10)/10
                label = index
                checkZero = True
            if col == 0.0 and checkZero: # A 0 after a 1 
                end = math.ceil((float(cindex)-0.01)*10)/10
                entry = {'event_onset': start, 'event_offset': end, 'event_label': str(label)}
                selDF = selDF.append(entry, ignore_index = True)
                checkZero = False
            if cindex == "5.00" and col == 1.0: # If it ends at the last time stamp
                end = float(cindex)
                entry = {'event_onset': start, 'event_offset': end, 'event_label': str(label)}
                selDF = selDF.append(entry, ignore_index = True)
    
    return selDF      

def f1_overall_framewise(O, T):
    TP = ((2*T-O) == 1).sum()         
    nr, ns = T.sum(), O.sum()

    prec = float(TP) / float(ns + eps)
    recall = float(TP) / float(nr + eps)
    f1 = 2 * prec * recall / (prec + recall + eps)

    return f1

def er_overall_framewise(O, T):
    FP = np.logical_and(T==0, O==1).sum(1)
    FN = np.logical_and(T==1, O==0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    nr = T.sum()
    ER = (S+D+I) / (nr + 0.0)

    return ER
################################################################

segment_len = 1
eps = np.finfo(float).eps

annotFolder = "../data/nips4b/annotations/"
annot1 = annotFolder+"010.csv"
annot2 = annotFolder+"020.csv"
mergedAnn = "../data/nips4b/mergedAnnotations/612_010.csv"

df1 = getCSV_DF(annot1)
df2 = getCSV_DF(annot2)
dfMerged = getCSV_DF(mergedAnn)

input1 = getInputMatrix(df1)
input2 = getInputMatrix(df2)
mInput = getInputMatrix(dfMerged)

altDF1 = from_InputDF_to_SELDF(input1)
altDF2 = from_InputDF_to_SELDF(input2)

print(f1_overall_framewise(input1.values, input1.values))
print(er_overall_framewise(input1.values, input1.values))