import scaper 
import os 

path_to_audio = "../data/nips4b/audio/"

soundscape_duration = 5.0
seed = 147

foreground_folder = path_to_audio, 'original'
background_folder = path_to_audio, 'background'
print(foreground_folder)
output_folder = "../data/nips4b/"

sc = scaper.Scaper(soundscape_duration, foreground_folder, background_folder,
                    random_state=seed)

sc.ref_db = -20
audio1 = path_to_audio+'original/train/nips4b_birds_trainfile006.wav'
audio2 = path_to_audio+'original/train/nips4b_birds_trainfile005.wav'

sc.add_event(label=('const', 'train'),
             source_file=('const', audio1),
             source_time=(0),
             event_time=(0),
             event_duration=(2.68),
             snr=('normal', 10, 3),
             pitch_shift=('uniform', -2, 2),
             time_stretch=('uniform', 0.8, 1.2))

sc.add_event(label=('const', 'train'),
             source_file=('const', audio2),
             source_time=(1.80),
             event_time=(3),
             event_duration=(1.35),
             snr=('normal', 10, 3),
             pitch_shift=('uniform', -2, 2),
             time_stretch=('uniform', 0.8, 1.2))

audiofile = os.path.join(output_folder, "testing.wav")
jamsfile = os.path.join(output_folder, "testing.jams")
txtfile = os.path.join(output_folder, "testing.txt")


sc.generate(audiofile, jamsfile,
            allow_repeated_label=True,
            allow_repeated_source=False,
            reverb=0.1,
            disable_sox_warnings=True,
            no_audio=False,
            txt_path=txtfile)