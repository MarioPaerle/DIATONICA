import os
from GridTokenizer2 import midi_to_segmented_pianorolls
import numpy as np
import joblib
import warnings
from tqdm import tqdm


FOLDER = "MuseScoreMIDIS/"
files = [f for f in os.listdir(FOLDER) if f.endswith(".mid")]
songs = []
lenght = 128
div = 2
for file in tqdm(files):
    for shift in range(0, 13):
        midi = midi_to_segmented_pianorolls(FOLDER + file, k=lenght, shift=shift, div=div)
        if isinstance(midi, np.ndarray):
            songs.append(midi[:, 20:100, :])

joblib.dump(songs, "MuseScoreMIDIS2Rolls1.pkl")