import pandas as pd 
from pydub import AudioSegment


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
    # Build name of new CSV
    csvName = getCombinedName(csvList, ".csv")

    # Create csv in folder
    targetCSV = outAnnotFolder+csvName
    concatCSV = open(targetCSV, "a+")

    # Fill the new CSV with the rows in the csvList 
    for i in range(len(csvList)):
        pos = positionList[i]
        annot = csvList[i]
        annotFile = open(csvList[i], "r")
        annotID = getFileID(annot)

        # Iterate CSV file
        for line in annotFile:
            elems = line.split(",")
            line = str(float(elems[0])+float(pos))+","+elems[1]+","+elems[2]
            concatCSV.write(line)
      
        annotFile.close()

    concatCSV.close()

    return csvName

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


############################ MAIN ############################

audioFolder = "../data/nips4b/audio/original/train/"
annotFolder = "../data/nips4b/annotations/"

outAudioFolder = "../data/nips4b/audio/o3/"
outAnnotFolder = "../data/nips4b/mergedAnnotations/"

stdDuration = 5

# FILES TO BE MERGED
csvList = [annotFolder+"125.csv", annotFolder+"297.csv", annotFolder+"377.csv"]
audioList = [audioFolder+"125.wav", audioFolder+"297.wav", audioFolder+"377.wav"]
positions = [0,0,0]

csvPath = concatSeveralCSV(csvList, posList)

# DATA FRAMES
df1 = getCSV_DF(annot1)
df2 = getCSV_DF(annot2)
dfConcat = getCSV_DF(concatPath)

print(df1)
print(df2)
print(dfConcat)

finalPath = outAudioFolder+concatCSV[:-4]+".wav"
file_handle = overlayAudio.export(finalPath, format="wav")

