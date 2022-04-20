import os 
import pandas as pd 
import numpy as np 
import math
import sed_eval
import dcase_util
import processing
import seaborn as sns
import matplotlib.pyplot as plt

def getCSV_DF(csvFile):
    columnNames = ['event_onset','event_duration', 'event_label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df

def getInputMatrix(csvDF, n_seps):
    ###########################################################
    # Gets DF from original csv (Start, Duration, Label) format
    # and the time separaton and transform it to binary matrix 
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

def getLabelList():
    speciesFile = "../data/nips4b/metadata/nips4b_birdchallenge_espece_list.csv"
    speciesDF = pd.read_csv(speciesFile) # Make it DataFrame
    labels = speciesDF["class name"].to_numpy()
    labels = np.delete(labels, 0) # Remove label 'Empty'

    return np.sort(labels)

def get_SEL_DF(otherDF):
    columnNames = ["event_onset", "event_offset", "event_label"]
    altDF = df = pd.DataFrame(columns = columnNames)

    for index, row in otherDF.iterrows():
        altDF.at[index, "event_onset"] = otherDF.at[index, "event_onset"]
        altDF.at[index, "event_offset"] = row.loc["event_onset"] + row.loc["event_duration"]
        altDF.at[index, "event_label"] = otherDF.at[index, "event_label"]

    return altDF

def from_InputDF_to_SELDF(inputDF):
    columnNames = ["event_onset", "event_offset", "event_label", "file", "scene_label"]
    selDF = df = pd.DataFrame(columns = columnNames)

    for index, row in inputDF.iterrows():
        checkZero = False
        for cindex, col in row.iteritems():
            if col == 1.0 and not checkZero: # First 1
                start = math.floor(float(cindex)*100000)/100000
                label = index
                checkZero = True
            if col == 0.0 and checkZero: # A 0 after a 1 
                end = math.ceil((float(cindex)-0.01)*100000)/100000
                entry = {'event_onset': start, 'event_offset': end, 'event_label': str(label), 'file':'', 'scene_label':'nature'}
                selDF = selDF.append(entry, ignore_index = True)
                checkZero = False
            if cindex == "5.00" and col == 1.0: # If it ends at the last time stamp
                end = float(cindex)
                entry = {'event_onset': start, 'event_offset': end, 'event_label': str(label), 'file':'', 'scene_label':'nature'}
                selDF = selDF.append(entry, ignore_index = True)
    
    return selDF      

def from_annotMatrix_to_annotDF(inputMatrix, labels):
    # Rows index 
    rowNames = labels

    # Column "names"
    n_frames = inputMatrix.shape[1]
    stamp = 5 / n_frames
    columnNames = np.arange(0, 5, stamp)
   
    # Create dataframe
    annotDF = pd.DataFrame(inputMatrix, columns=columnNames)
    annotDF.index = rowNames

    return annotDF

def f1_score_all(output, truth):
    # output = processing.output_to_binary(output, bin_thres).T
    # output = output.reshape(output.shape[-2], output.shape[-1])
    # truth = processing.output_to_binary(truth, bin_thres).T

    TP = ((2*truth-output) == 1).sum()  
    FP = np.logical_and(truth==0, output==1).sum()
    FN = np.logical_and(truth==1, output==0).sum()       
    TN = np.logical_and(truth==0, output==0).sum()

    # print("TP, TN, FP, FN: ", TP, TN, FP, FN)

    prec = float(TP) / float(TP+FP + eps)
    recall = float(TP) / float(TP+FN + eps)
    f1 = 2 * prec * recall / (prec + recall + eps)

    return f1

def er_all(output, truth):
    # output = processing.output_to_binary(output, bin_thres).T
    # output = output.reshape(output.shape[-2], output.shape[-1])
    # truth = processing.output_to_binary(truth, bin_thres).T

    FP = np.logical_and(truth==0, output==1).sum()
    FN = np.logical_and(truth==1, output==0).sum()

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    nr = truth.sum()
    ER = (S+D+I) / (nr + 0.0)

    return ER

def get_score(output, truth):
    f1 = f1_score_all(output, truth)
    er = er_all(output, truth)

    return f1, er

def sed_eval_scores(pred, truth, labels): # Prediction and truth must be already reshaped and re-transposed
    # Transform to binary matrix Dataframe
    pred_DF = from_annotMatrix_to_annotDF(pred, labels)
    truth_DF = from_annotMatrix_to_annotDF(truth, labels)
    
    # Transform reference matrix (binary) into event list 
    pred_events = from_InputDF_to_SELDF(pred_DF)
    truth_events = from_InputDF_to_SELDF(truth_DF)

    # Transform DataFrame to dictionary
    pred_dict = pred_events.to_dict('records')
    truth_dict = truth_events.to_dict('records')
       
    # Get MetaDataContainers of both matrices
    pred_event_list = dcase_util.containers.MetaDataContainer(pred_dict)
    truth_event_list = dcase_util.containers.MetaDataContainer(truth_dict)

    # Get Segment and Event metrics
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=labels,
        time_resolution=1.0
    )
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=labels,
        t_collar=0.250
    )

    # Evaluate 
    segment_based_metrics.evaluate(
        reference_event_list=truth_event_list,
        estimated_event_list=pred_event_list
    )

    event_based_metrics.evaluate(
        reference_event_list=truth_event_list,
        estimated_event_list=pred_event_list
    )

    # Overall metrics 
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    overall_event_based_metrics = event_based_metrics.results_overall_metrics()

    return overall_event_based_metrics, overall_segment_based_metrics

def plotPredTruth(pred, pred_nothres, truth, labels):
    # Transform to DF
    pred = from_annotMatrix_to_annotDF(pred, labels)
    pred_nothres = from_annotMatrix_to_annotDF(pred_nothres, labels)
    pred_nothres = pred_nothres.to_numpy()
    truth = from_annotMatrix_to_annotDF(truth, labels)
    

    # Plot
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.set_title("Prediction binary")
    ax2.set_title("Prediction w/o threshold")
    ax3.set_title("Truth")
    fig.subplots_adjust(wspace=0.01)

    sns.heatmap(pred, cmap="viridis", ax=ax1, cbar=False)
    sns.heatmap(pred_nothres, cmap="viridis", ax=ax2, cbar=False)
    sns.heatmap(truth, cmap="viridis", ax=ax3, cbar=False)

    fig.subplots_adjust(wspace=0.001)
    plt.show()

################################################################
bin_thres = 0.5
segment_len = 1
eps = np.finfo(float).eps
stdDuration = 5
sep = 431

# annotFolder = "../data/nips4b/annotations/"
# annot1 = annotFolder+"010.csv"
# annot2 = annotFolder+"020.csv"
# mergedAnn = "../data/nips4b/mergedAnnotations/612_010.csv"

# df1 = getCSV_DF(annot1)
# df2 = getCSV_DF(annot2)
# # dfMerged = getCSV_DF(mergedAnn)

# input1 = getInputMatrix(df1, sep)
# input2 = getInputMatrix(df2, sep)
# # mInput = getInputMatrix(dfMerged, sep)

# inDF1 = from_annotMatrix_to_annotDF(input1)

# altDF1 = from_InputDF_to_SELDF(inDF1)

# altDF2 = from_InputDF_to_SELDF(input2)

# f1_ev = sed_eval_scores(input1,input1)[1]['f_measure']['f_measure']
# er_ev = sed_eval_scores(input1,input1)[1]['error_rate']['error_rate']

# processing.plotPredTruth(from_annotMatrix_to_annotDF(input1),from_annotMatrix_to_annotDF(input1))
# print(f1_score_all(test1, test2))
# print(er_all(test1, test2))