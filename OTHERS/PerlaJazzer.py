from intomido import encodings, composers
import mido
import pretty_midi
import os
import matplotlib.pyplot as plt
import pypianoroll
import numpy as np
import random as rd
from intomido.functions import pm_swing

folder = os.listdir("MuseScoreMIDIS")
file = pretty_midi.PrettyMIDI("MuseScoreMIDIS/chpn_op10_e05_format0.mid")

file.instruments = file.instruments[:1]
file.instruments[0].notes = [k for k in file.instruments[0].notes if k.pitch > 70]
for note in file.instruments[0].notes:
    note.pitch += rd.randint(0, 5)//4*int(np.sign(rd.random()))
    note.velocity = min(max(50, note.velocity + (rd.randint(0, 60) - 15)), 127)

pm_swing(file)
# file.get_piano_roll()
# pianoroll = file.get_piano_roll()[:, :1000]
file.write("ROnly2.mid")

