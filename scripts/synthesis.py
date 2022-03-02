from pydub import AudioSegment
import pandas as pd
import shutil

########################### METHODS ###########################

# Concatenate CSV2 and CSV1
def getFileID(fileName):
    fileID = fileName[:-4] # Remove .csv
    fileID = fileID[-3:] # Remove the rest, keep only numbers(ID)
    return fileID

def concatCSV(csv1, csv2, position1, position2):
    id1 = getFileID(csv1)
    id2 = getFileID(csv2)
    
    # Create empty CSV
    csvName = id1+"_"+id2+".csv"
  
    original = csv1
    target = outAnnotFolder+csvName

    #shutil.copyfile(original, target)

    concatCSV = open(target, "a+")

    # Fill final csv
    file1 = open(csv1, "r")
    file2 = open(csv2, "r")

    for line in file1:
        elems = line.split(",")
        print(elems[0])
        line = str(float(elems[0])+float(position1))+","+elems[1]+","+elems[2]
        
        concatCSV.write(line)
    for line in file2:
        # Change start time depending on position 
        elems = line.split(",")
        line = str(float(elems[0])+float(position2))+","+elems[1]+","+elems[2]
        
        concatCSV.write(line)

    # Close files
    file1.close()
    file2.close()
    concatCSV.close()

    return csvName

def getCSV_DF(csvFile):
    columnNames = ['Start','Duration', 'Label']
    df = pd.read_csv(csvFile, names = columnNames)
    return df


############################ MAIN ############################

audioFolder = "../data/nips4b/audio/original/train/"
annotFolder = "../data/nips4b/annotations/"

outAudioFolder = "../data/nips4b/audio/o3/"
outAnnotFolder = "../data/nips4b/mergedAnnotations/"

stdDuration = 10

sound1 = AudioSegment.from_file(audioFolder+"149.wav", format="wav")
sound2 = AudioSegment.from_file(audioFolder+"012.wav", format="wav")
emptyAudio = AudioSegment.silent(duration=1000*stdDuration)

#Combine AUDIOS
pos1 = 2
pos2 = 3
overlayAudio = emptyAudio.overlay(sound1, position=pos1)
overlayAudio = overlayAudio.overlay(sound2, position=pos2)

# Combine ANNOTATIONS
annot1 = annotFolder+"149.csv"
annot2 = annotFolder+"012.csv"
concatCSV = concatCSV(annot1, annot2, pos1, pos2)
concatPath = outAnnotFolder+concatCSV

# DATA FRAMES
df1 = getCSV_DF(annot1)
df2 = getCSV_DF(annot2)
dfConcat = getCSV_DF(concatPath)

print(df1)
print(df2)
print(dfConcat)

finalPath = outAudioFolder+concatCSV[:-4]+".wav"
file_handle = overlayAudio.export(finalPath, format="wav")

