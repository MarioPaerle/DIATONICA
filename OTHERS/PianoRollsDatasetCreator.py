import os
from GridTokenizer2 import midi_to_segmented_pianorolls
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from intomido.composers import Pianoroll

FOLDER = "ALLMIDIS/"
files = [f for f in os.listdir(FOLDER) if f.endswith(".midi")]
songs = []
lenght = 512
# div = 4
for file in tqdm(files):
    for shift in range(0, 12):
        shift = shift - 6
        for dd in range(0, 1):
            dd = 4
            if 'bwv' in file:
                dd = dd*4
            midi = midi_to_segmented_pianorolls(FOLDER + file, k=lenght, shift=shift, div=dd)
            if isinstance(midi, list) or midi.max() == 0 or midi.mean() < 0.002:
                print('killed zeros')
                continue
            for i in range(midi.shape[0] - 1):
                if midi[i].mean() < 0.002:
                    midi[i] = midi[i+1]
                    print('zeros!')
                    """plt.imshow(midi[i, 20:100, :])
                    plt.show()"""

            midi[midi > 0] = 1
            """plt.imshow(midi[0, 20:100, :])
            plt.show()"""
            """pianoroll = Pianoroll(64, 32)
            pianoroll.grid = midi[0]
            pianoroll.plot()
            pianoroll.play()
            break"""
            if isinstance(midi, np.ndarray):
                songs.append(midi[:, 20:100, :])

joblib.dump(songs, "Maestro_Rolls1.pkl", compress=9)