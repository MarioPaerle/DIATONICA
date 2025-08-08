import os
from GridTokenizer2 import midi_to_segmented_pianorolls
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from intomido.composers import Pianoroll
from tqdm import tqdm
import json

files = json.load(open("maestro-v2.0.0/maestro-v2.0.0.json"))
FOLDER = 'maestro-v2.0.0'
lenght = 1024
songs_train = []
meta_train = []
songs_test = []
meta_test = []
songs_val = []
meta_val = []
for idx, file in enumerate(tqdm(files)):
    for shift in range(0, 6):
        composer = file['canonical_composer']
        """if "Chopin" not in composer and "Liszt" not in composer and "Schubert" not in composer and "Rachmaninoff" not in composer:
            break"""
        filename = file['midi_filename']
        split = file['split']
        shift = shift - 3
        for dd in range(0, 1):
            dd = 1
            if 'bwv' in file:
                dd = dd*4
            midi = midi_to_segmented_pianorolls(FOLDER + '/' + filename, k=lenght, shift=shift, div=dd)
            if isinstance(midi, list) or midi.max() == 0 or midi.mean() < 0.0005:
                print('killed zeros')
                continue
            for i in range(midi.shape[0] - 1):
                if midi[i].mean() < 0.002:
                    midi[i] = midi[i+1]
                    print('zeros!')
                    """plt.imshow(midi[i, 20:100, :])
                    plt.show()"""

            """midi[midi > 0] = 1
            plt.imshow(midi[0, 20:100, :])
            plt.show()"""
            """pianoroll = Pianoroll(64, 32)
            pianoroll.grid = midi[0]
            pianoroll.plot()
            pianoroll.play()
            break"""
            data = midi[:, 20:100, :]
            meta = composer, file["canonical_title"]
            if isinstance(midi, np.ndarray):
                if split == 'train':
                    songs_train.append(data)
                    meta_train.append(meta)
                if split == 'test':
                    songs_test.append(data)
                    meta_test.append(meta)
                if split == 'val':
                    songs_val.append(data)
                    meta_val.append(meta)

joblib.dump(songs_train, "Maestro_Rolls_full2_train.pkl", compress=9)
joblib.dump(meta_train, "Maestro_Rolls_full2_mtrain.pkl", compress=9)
joblib.dump(songs_test, "Maestro_Rolls_full2_test.pkl", compress=9)
joblib.dump(meta_test, "Maestro_Rolls_full2_mtest.pkl", compress=9)
joblib.dump(songs_val, "Maestro_Rolls_full2_val.pkl", compress=9)
joblib.dump(meta_val, "Maestro_Rolls_full2_mval.pkl", compress=9)
