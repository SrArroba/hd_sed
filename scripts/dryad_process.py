import os 
import pandas as pd 
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import sed_eval
import seaborn as sns
from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise, TimeStretch, PitchShift, Shift
from sklearn import preprocessing
import random

########################################### METHODS ############################################################
def chooseRandomFiles(polyphony):
    #### Return a list of file IDs with length between 2 and the desired overlap 

    nFiles = random.randint(3, polyphony+3) # Number of files to merge
    chosenIDs = [] # List of chosen IDS to merge

    for i in range(nFiles):
        rdmIndex = random.randint(0, len(annotAll)-1) # Choose random file from annotation list
        
        if(rdmIndex in chosenIDs): # Check that the file is not already considered
            continue
        
        chosenIDs.append(rdmIndex)
    return chosenIDs

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

def del_low_spec(t):
    speciesCount = np.zeros(shape=len(orig_speciesList))
    annotFolds = ["../data/dryad/annotation/Recording_1/", "../data/dryad/annotation/Recording_2/", 
                "../data/dryad/annotation/Recording_3/","../data/dryad/annotation/Recording_4/"]
    n_activ = []
    # Obtain number of activations for each species
    for fold in annotFolds:
        listFiles = os.listdir(fold)
        for f in listFiles:
            with open(fold+f, "r") as rf: 
                lines = rf.readlines()
                n_activ.append(len(lines))
                lines.pop(0) # Remove header
                for line in lines:
                    spc = line[-5:].replace("\n", "")
                    speciesCount[orig_speciesList.index(spc)] += 1
    # print("Mean number of activations in the files: ", np.mean(n_activ))
    # print("Total number of activations: ", int(np.sum(speciesCount))) 
    
    # Return a species with less than t activations
    removeSpec = []
    # print("Density list: ")
    for i in range(len(speciesCount)):
        # print(speciesList[i], " --> ", int(speciesCount[i]))
        if(int(speciesCount[i]) < t):
            removeSpec.append(orig_speciesList[i])
    
    return removeSpec

def generateDataset(n_files, polyphony):
    inFeat = []
    inAnnot = []
    count = 0
    print("GENERATING DATASET of ", n_files," files and polyphony of ", polyphony)
    while(count != n_files): # Iterate until reaching desired amount of files
        # Choose files
        chosenIndex = chooseRandomFiles(polyphony)

        ##### MERGE AUDIOS #####
        finalAudio, toPlay = mergeAudios(chosenIndex)

        ##### MERGE ANNOTS #####
        finalAnnot = mergeAnnots(chosenIndex)
        # finalAnnot = finalAnnot.sort_values('event_label')

        ##### OBTAIN FEATURE and BINARY MATRIX #####
        feat = getMelSpectrogram(finalAudio)
        inputMatrix = getInputMatrix(finalAnnot, feat.shape[1])

        ##### CHECK POLYPHONY #####
        finalOv = countOverlap(inputMatrix)
        
        ##### CHECK IF MATCHING DESIRED OVERLAPPING #####
        if(finalOv <= polyphony): # Take only files with less or equal polyphony as desired
            count += 1
            print("Generating dataset: {} out of {} ({} %)".format(count, n_files, (100*count/n_files)), end='\r')

            ##### PRINTS #####
            # plotSpec_and_Annot(feat, inputMatrix)
            # play(toPlay)

            ##### NORMALIZE FEATURE #####
            feat = normalize_data(feat.T)
            
            ##### APPEND TO DATASET LIST ###### 
            feat.astype(np.float32)
            inputMatrix.astype(np.float32)
            inFeat.append(feat)
            inAnnot.append(inputMatrix.T)

    inFeat = np.array(inFeat)
    inAnnot = np.array(inAnnot)

    print("In shape: ", inFeat[0].shape, "(", type(inFeat[0]), ")")
    print("Out shape: ", inAnnot[0].shape, "(", type(inAnnot[0]), ")")

    return inFeat, inAnnot

def get_all_annot_and_feat():
    #######################################################
    # Create all features and corresponding annotations   #
    # from all the recordings                             #
    #######################################################
    audioAll = []
    annotAll = []
    for i in range(len(audioFolds)):
        # Obtain audio and annot files (sorted)
        audioFiles = os.listdir(audioFolds[i])
        annotFiles = os.listdir(annotFolds[i])
        audioFiles.sort()
        annotFiles.sort()

        # Iterate each ID (pair of audoo/annot files)
        for j in range(len(audioFiles)):
            audioPath = audioFolds[i]+audioFiles[j]
            annotPath = annotFolds[i]+annotFiles[j]
            print("Processing file... ", audioPath, end='\r')
            # Split DataFrame
            df = getTXT_DF(annotPath)
            df_list, valid_starts = separate_annot(df, clip_win, clip_hop)

            # Split audio file 
            specList = splitAudio(audioPath, valid_starts)

            # Append to major list
            for k in range(len(df_list)):     
                # # Obtain Binary (presence) Matrix
                # inputMatrix = getInputMatrix(df_list[k], specList[k].shape[1])                

                # Append
                annotAll.append(df_list[k])
                audioAll.append(specList[k])

    return audioAll, annotAll

def getInputMatrix(df, n_seps):
    ###########################################################
    # Gets DF from original csv (Start, End, Label) format
    # and the time separation from the feature 
    # and transform it to binary matrix that will be fed to the 
    # model as output/annotations 
    # --> Output format (n_classes, time_stamps)
    ###########################################################

    # Determines how many divisions are in the time scale 
    stamp = clip_win/n_seps

    # Tranform to a records dictionary - E.g. {onset: "", offset: "", label: ""}
    dictEvents = df.to_dict('records')   

    # Obtain binary matrix 
    mat = sed_eval.util.event_roll.event_list_to_event_roll(dictEvents, speciesList, stamp)

    mat = np.transpose(mat) # Transpose
   
    # Check if is bigger than the standard length:
    if(mat.shape[1] > n_seps):
        mat = mat[:, 0:n_seps]

    # Check if is smaller than the standard length:
    if(mat.shape[1] < n_seps):
        colMissing = n_seps - mat.shape[1]
        matZeros = np.zeros((len(speciesList), colMissing))
        
        # Concatenate original matrix with the submatrix of 0s
        mat = np.concatenate((mat, matZeros), axis=1)

    # Print info (can be commented)
    # print(mat, " (", mat.shape, ")", type(mat))
    
    return mat

def getMelSpectrogram(audioSegment):
    # Get samples and transform to np array
    # samples = audioSegment.get_array_of_samples()
    # arr = np.array(samples).astype(np.float32)

    # Obtain librosa mel spectrogram
    S = librosa.feature.melspectrogram(y=audioSegment, sr=sr,  power=1, win_length=win_len, hop_length=hop_len, n_mels=n_mels)

    # Power to decibels
    S_dB = librosa.amplitude_to_db(S, ref=np.max, top_db=85, amin=1e-05)

    #S = np.delete(S , -1, axis=1)
    # Plot spectrogram (can be commmented)
    # plotSpec(S_dB)    
    
    return S_dB

def getTXT_DF(csvFile):
    df = pd.read_csv(csvFile, sep='\t')

    # Remove unnecessary columns
    del df["Selection"]
    del df["View"]
    del df["Channel"]
    del df["Low Freq (Hz)"]
    del df["High Freq (Hz)"]

    # Change column names
    df = df.set_axis(['event_onset', 'event_offset', 'event_label'], axis=1, inplace=False)

    return df

def mergeAnnots(annotList): # Input: Index in 
    # Create a list of DataFrames 
    frames = []
    for i in annotList:
        frames.append(annotAll[i])

    # Concatenate list of DF into a single DF
    result = pd.concat(frames, ignore_index=True) # Ignore index = continues index list without starting from 0 again

    return result

def mergeAudios(audioList): # Merge from audio list and generate an AudioSegment object
    # Generate empty audio file
    finalAudio = AudioSegment.silent(duration=1000*clip_win)

    # Merge every recording in the list into final audio
    for i in audioList:
        finalAudio = finalAudio.overlay(audioAll[i], position=0)

    toPlay = finalAudio

    samples = finalAudio.get_array_of_samples()
    finalAudio = np.array(samples).astype(np.float32)
    
    # Data augmentation
    augment = Compose([
        #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.15, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    # finalAudio = augment(samples=finalAudio, sample_rate=sr)

    return finalAudio, toPlay

def normalize_data(feature):
    stdScal = preprocessing.StandardScaler()
    feature = stdScal.fit_transform(feature)
    
    return feature

def separate_annot(dataframe, clip_win, clip_hop):
    # Set sections start-end points
    start_pts = np.arange(0, dur-clip_hop, clip_hop)
    end_pts = np.arange(clip_win, dur+clip_hop, clip_hop)
   
    n_sections = len(start_pts)

    cut_row = []
    df_list = []
    df_red_list = []
    valid_starts = []
    for sec_id in range(n_sections):
        mini_df = pd.DataFrame(columns=['event_onset', 'event_offset', 'event_label'])
        red_df = pd.DataFrame(columns=['event_onset', 'event_offset', 'event_label'])
        # Check if previous 'overflow'
        if(len(cut_row) != 0 and row['event_label'] in speciesList):
            for r in cut_row:
                mini_df = mini_df.append(r, ignore_index=True)
                # Reduced DF
                red_start = r['event_onset'] - start_pts[sec_id]
                red_end = r['event_offset'] - start_pts[sec_id]
                red_df = red_df.append({'event_onset': red_start, 'event_offset': red_end, 'event_label': row['event_label']}, ignore_index=True)
        cut_row = []
        for index, row in dataframe.iterrows():
            # Check starting and ending times 
            if(row['event_label'] in speciesList): # Use only acceptable species
                if(row['event_onset'] >= start_pts[sec_id]):
                    if(row['event_offset'] <= end_pts[sec_id]):
                        mini_df = mini_df.append(row, ignore_index=True)
                        # Reduced df: From 0 to clip_win
                        red_start = row['event_onset'] - start_pts[sec_id]
                        red_end = row['event_offset'] - start_pts[sec_id]
                        red_df = red_df.append({'event_onset': red_start, 'event_offset': red_end, 'event_label': row['event_label']}, ignore_index=True)
                    
                    elif(row['event_offset'] > end_pts[sec_id] and row['event_onset'] < end_pts[sec_id]):
                        newRow = {'event_onset': row['event_onset'], 'event_offset': end_pts[sec_id], 'event_label': row['event_label']}
                        mini_df = mini_df.append(newRow, ignore_index=True)
                        cut_row.append({'event_onset': end_pts[sec_id], 'event_offset': row['event_offset'], 'event_label': row['event_label']})
                        # Reduced DF
                        red_start = newRow['event_onset'] - start_pts[sec_id]
                        red_end = newRow['event_offset'] - start_pts[sec_id]
                        red_df = red_df.append({'event_onset': red_start, 'event_offset': red_end, 'event_label': row['event_label']}, ignore_index=True)
               
        # Add to final DF list if not empty
        if(mini_df.empty == False): 
            df_list.append(mini_df)
            df_red_list.append(red_df)
            valid_starts.append(start_pts[sec_id])
            
            # print("\nMINIDF in ", start_pts[sec_id], " to ", end_pts[sec_id])
            # print(mini_df)
            # print("-----")
            # print(red_df)

    valid_starts.sort()
    return df_red_list, valid_starts
 
def plotAnnotMatrix(annot):

    fig, ax = plt.subplots(1, 1)

    #Display with vertical lines on the heat map (right side)
    sns.heatmap(annot, vmin=0, vmax=5, ax=ax)
    # ax.axvline(x=1, linewidth=2, color="w")
    # ax.axvline(x=2, linewidth=2, color="w")
    # ax.axvline(x=3, linewidth=2, color="w")
    # ax.axvline(x=4, linewidth=2, color="w")
    plt.show()

def plotSpec(S):
    fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

def plotSpec_and_Annot(S, annot):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Spec
    img = librosa.display.specshow(S, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax1)
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    ax1.set(title='Mel-frequency spectrogram')
    
    # Annot
    sns.heatmap(annot, vmin=0, vmax=5, ax=ax2)
    # ax.axvline(x=1, linewidth=2, color="w")
    # ax.axvline(x=2, linewidth=2, color="w")
    # ax.axvline(x=3, linewidth=2, color="w")
    # ax.axvline(x=4, linewidth=2, color="w")
    plt.show()

def purge_species():
    species = []
    for spec in orig_speciesList:
        if(spec not in rmSpec):
            species.append(spec)

    return species

def splitAudio(filePath, start_pts):
    ##############################################
    # Separate audio file into clips of clip_win #
    # and obtain the corresponding spectrogram   #
    ##############################################
    feats = []
    audio = AudioSegment.from_wav(filePath)

    for pt in start_pts:
        t1 = pt * 1000              # Stored in ms
        t2 = (pt+clip_win) * 1000   # Stored in ms

        clip = audio[t1:t2]
        
        # Obtain feature (e.g. spectrogram)
        #feat = getMelSpectrogram(clip)
        
        # plotSpec(feat)
        # print(feat.shape)
        
        feats.append(clip)

    return feats
    # test1.export(audioFolds[2]+'test1.wav', format="wav")
    # test2.export(audioFolds[2]+'test2.wav', format="wav")

    
###############################################################################################################

# Folders and paths 
audioFold = "../data/dryad/audio/Recording_1/"
audioFolds = ["../data/dryad/audio/Recording_3/"]#, "../data/dryad/audio/Recording_2/", 
                # "../data/dryad/audio/Recording_3/","../data/dryad/audio/Recording_4/"]

annotFold = "../data/dryad/annotation/Recording_1/"
annotFolds = ["../data/dryad/annotation/Recording_3/"]#, "../data/dryad/annotation/Recording_2/", 
                # "../data/dryad/annotation/Recording_3/","../data/dryad/annotation/Recording_4/"]

audioList = os.listdir(audioFold)
annotList = os.listdir(annotFold)
audioList.sort()
annotList.sort()

# COMPLETE LIST OF SPECIES 
orig_speciesList = ['AMCR', 'AMGO' , 'AMRE', 'AMRO', 'BAOR', 'BAWW', 'BBWA', 'BCCH', 'BGGN', 'BHCO', 'BHVI', 
                    'BLJA', 'BRCR', 'BTNW', 'BWWA', 'CANG', 'CARW', 'CEDW', 'CORA', 'COYE', 'CSWA', 'DOWO', 
                    'EATO', 'EAWP', 'HAWO', 'HETH', 'HOWA', 'KEWA', 'LOWA', 'NAWA', 'NOCA', 'NOFL', 'OVEN', 
                    'PIWO', 'RBGR', 'RBWO', 'RCKI', 'REVI', 'RSHA',	'RWBL', 'SCTA',	'SWTH', 'TUTI',	'VEER', 
                    'WBNU', 'WITU',	'WOTH', 'YBCU']

# Species that have very few activations - TO BE REMOVED FROM CONSIDERATION 
time_thres = 100 # Minimum number of activations to be valid
rmSpec = del_low_spec(time_thres)
speciesList = purge_species()

print("ELIM:", len(rmSpec))
print(rmSpec)
print("REMAINING: ", len(speciesList))
print(speciesList)

# Audio length properties
dur = 300 # In seconds (=5min)
clip_win = 5
clip_hop = 2.5

# Spectrogram parameters
win_len = 1024
hop_len = 512
sr = 32000
n_mels = 128

# Generate features and annotations
# a, b = separate_annot(getTXT_DF(annotFold+annotList[2]), clip_win, clip_hop)
# print(getTXT_DF(annotFold+annotList[2]))
# splitAudio(audioFold+audioList[2], b)

# Create all clips and annotations from oiginal files
audioAll, annotAll = get_all_annot_and_feat()
print("FINAL LENGTH: ", len(audioAll), len(annotAll))
# feats, annots = generateDataset(100, 3)

