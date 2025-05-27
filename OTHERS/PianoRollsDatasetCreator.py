import os
from GridTokenizer2 import midi_to_segmented_pianorolls
import numpy as np
import joblib
import warnings
from tqdm import tqdm


FOLDER = "MuseScoreMIDIS2/"
files = [f for f in os.listdir(FOLDER) if f.endswith(".mid")]
songs = []
lenght = 256
# div = 4
for file in tqdm(files):
    for shift in range(0, 13):
        for dd in range(1, 5):
            midi = midi_to_segmented_pianorolls(FOLDER + file, k=lenght, shift=shift, div=dd)
            if isinstance(midi, np.ndarray):
                songs.append(midi[:, 20:100, :])
                print(midi.shape)

joblib.dump(songs, "MS2_big1_dataset.pkl", compress=9)