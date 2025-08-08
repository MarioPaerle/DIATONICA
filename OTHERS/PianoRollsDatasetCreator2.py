import os
from GridTokenizer2 import midi_to_segmented_pianorolls
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from intomido.composers import Pianoroll
from tqdm import tqdm
import json

files = os.listdir(r"C:\Users\mario\OneDrive\Documenti\GitHub\DIATONICA\OTHERS\MuseScoreMIDIS2")
FOLDER = r"C:\Users\mario\OneDrive\Documenti\GitHub\DIATONICA\OTHERS\MuseScoreMIDIS2"
lenght = 1024
songs_train = []
meta_train = []
songs_test = []
meta_test = []
songs_val = []
meta_val = []
for idx, file in enumerate(tqdm(files)):

    for shift in range(0, 6):
        shift = shift - 3
        for dd in range(0, 1):
            dd = 1

            midi = midi_to_segmented_pianorolls(FOLDER + '/' + file, k=lenght, shift=shift, div=dd)
            if isinstance(midi, list) or midi.max() == 0 or midi.mean() < 0.0005:
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
            plt.show()
            pianoroll = Pianoroll(64, 32)
            pianoroll.grid = midi[0]
            pianoroll.plot()
            pianoroll.play(1/2)
            break"""
            data = midi[:, 20:100, :]
            if idx % 6 == 0:
                songs_test.append(data)
                meta_test.append(file)
            else:
                songs_train.append(data)
                meta_train.append(file)


joblib.dump(songs_train, "MuseScoreMidis2_Augmented_train.pkl", compress=9)
joblib.dump(songs_test, "MuseScoreMidis2_Augmented_test.pkl", compress=9)
joblib.dump(meta_train, "MuseScoreMidis2_Augmentedm_train.pkl", compress=9)
joblib.dump(meta_test, "MuseScoreMidis2_Augmented_mtest.pkl", compress=9)
