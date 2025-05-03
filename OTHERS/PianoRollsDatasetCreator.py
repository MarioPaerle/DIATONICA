import os
import pretty_midi
import numpy as np
import joblib

FOLDER = "MuseScoreMIDIS/"
files = [f for f in os.listdir(FOLDER) if f.endswith(".mid")]
print(files)