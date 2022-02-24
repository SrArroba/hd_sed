import os
import scaper
import numpy as np

# Folder
outfolder = '../data/nips4b/audio/overlap/'

# Foreground and Background
fg_folder = '../data/nips4b/audio/original/train/'
bg_folder = '../data/nips4b/audio/background/'

n_soundscapes = 10
ref_db = -50
duration = 10.0

# A random number between these two 
# is generated to create soundscapes
min_events = 1
max_events = 3

# Properties
event_time_dist = 'truncnorm'
event_time_mean = 5.0
event_time_std = 2.0
event_time_min = 0.0
event_time_max = 10.0

source_time_dist = 'const'
source_time = 0.0

event_duration_dist = 'uniform'
event_duration_min = 0.5
event_duration_max = 4.0

snr_dist = 'uniform'
snr_min = 6
snr_max = 30

pitch_dist = 'uniform'
pitch_min = -3.0
pitch_max = 3.0

time_stretch_dist = 'uniform'
time_stretch_min = 0.8
time_stretch_max = 1.2

# Random seed
seed = 147

# Scaper object 
sc = scaper.Scaper( duration, fg_folder, bg_folder, 
                    random_state = seed)
sc.protected_labels = []
sc.ref_db = ref_db

# Create specified soundscapes with truncated normal
# distribution of start times
for n in range(n_soundscapes):
    print("Creating soundscape: {:d}/{:d}".format(n+1, n_soundscapes))
    
    #sc.reset_fg_spec()
    #sc.reset_bg_spec()

    # Could add background here if wanted
    # sc.add_background...

    # Random number of events:
    n_events = np.random.randint(min_events, max_events)

    for _ in range(n_events):
        sc.add_event(label=('choose', []), 
                     source_file=('choose', []),
                     source_time=(source_time_dist, source_time),
                     event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max),
                     event_duration=(event_time_dist, event_duration_min, event_duration_max),
                     snr=(snr_dist, snr_min, snr_max),
                     pitch_shift=(pitch_dist, pitch_min, pitch_max),
                     time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))

    audiofile = os.path.join(outfolder, "soundscape_unimodal{:d}.wav".format(n))
    jamsfile = os.path.join(outfolder, "soundscape_unimodal{:d}.jams".format(n))
    txtfile = os.path.join(outfolder, "soundscape_unimodal{:d}.txt".format(n))

    sc.generate(audiofile, jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=False,
                reverb=0.1,
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile)