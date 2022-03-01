from pydub import AudioSegment

audioFolder = "../data/nips4b/audio/original/train/"
annotFolder = "../data/nips4b/annotations/"
outFolder = "../data/nips4b/audio/o3/"


sound1 = AudioSegment.from_file(audioFolder+"nips4b_birds_trainfile149.wav", format="wav")
sound2 = AudioSegment.from_file(audioFolder+"nips4b_birds_trainfile241.wav", format="wav")

combined = sound1 + sound2
sound1 = sound1 + 6
overlay = sound1.overlay(sound2, position=0)


annot1 = annotFolder+"annotation_train149.csv"
annot2 = annotFolder+"annotation_train241.csv"

file_handle = combined.export(outFolder+"combined.wav", format="wav")
file_handle = overlay.export(outFolder+"overlay.wav", format="wav")

