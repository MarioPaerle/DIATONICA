import joblib
from intomido.functions import *
import matplotlib.pyplot as plt

filename = "VAEpred9.pkl"
name = filename.split('.')[0]
file = joblib.load(filename)
print(file)
t = 0.3
file[file > t] = 1
file[file <= t] = 0
print(file.shape)

plt.imshow(file)
plt.show()

midi = multi_hot_to_midi(file.T)
midi.write(f'{name}.mid')
audio = midi_to_audio(midi, outputfile=f'{name}.wav')