import numpy as np

from intomido.functions import *
import os
import joblib

midis = []
lenn = 250
skip = 2
files = os.listdir("datas")
for file in files:
    if file.endswith(".midi"):
        midis.append(midi_to_numpy("datas/" + file))

# HOW TO EASILY LISTEN TO A MIDI
if False:
    plotmidi(midis[2])
    audio = midi_to_audio("datas/" + files[2])

    ipd.display(ipd.Audio(audio, rate=44100))  # THIS FOR THE NOTEBOOKS

    scaled_audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write("output.wav", 44100, scaled_audio)

reshaped = []
bn = []
for midi in midis:
    midi = midi[:, 100::skip]
    bins_number = midi.shape[1] // lenn
    bn.append(bins_number)

    pieces = np.uint8(midi[:, :bins_number*lenn].reshape(128, bins_number*lenn // lenn, lenn)).transpose(1, 0, 2)

    for piece in pieces:
        reshaped.append(piece)


reshaped = np.array(reshaped).astype(np.uint8)


print(reshaped.shape)

datas = {
    'onhot': reshaped,
}

joblib.dump(datas, 'datas/maestro3_comp.pkl', compress=("gzip", 3))
print('done')