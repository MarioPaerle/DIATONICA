import joblib
from intomido.functions import *
import matplotlib.pyplot as plt

filename = "pred5.pkl"
file = joblib.load(filename)[0]
print(file)
t = 0.2
file[file > t] = 1
file[file <= t] = 0
print(file.shape)

plt.imshow(file)
plt.show()

midi = multi_hot_to_midi(file.T)
audio = midi_to_audio(midi, outputfile='output4.wav')