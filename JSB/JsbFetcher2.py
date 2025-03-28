import numpy as np
import json
from intomido.functions import *
import matplotlib.pyplot as plt
import joblib
S = 1  # coprimo con 12

with open("datas/chorale1.json", 'r') as f:
    chorale1 = json.load(f)

tipo = 'test'
name = 'harmon'

def midi_to_s(X):
    X = np.where(X % 2 == 1, X - (S-1), X)
    return X


def s_to_midi(X):
    X = np.where(X % 2 == 1, X + (S-1), X)
    return X



minimus = min([len(c) for c in chorale1[tipo]])
notes = []
for c in chorale1[tipo]:
    try:
        c = np.array(c)[:100]
        notes.append(c)
    except:
        pass

notes = np.array(notes)
notes = midi_to_s(notes)




########################################################################################################################
transposed = []
for i in range(12):
    transposed.append(notes + i-6)



transposed = np.array(transposed)
transposed = transposed.reshape(12*transposed.shape[1], transposed.shape[2], transposed.shape[3])

melodies = transposed[:, :, :1]
harmonies = transposed[:, :, 1:]

transposed = mod_to_midi_representation(transposed, 1)
transposed = np.array(transposed)
melodies = np.array(mod_to_midi_representation(melodies, 1))
harmonies = np.array(mod_to_midi_representation(harmonies, 1))



X = transposed.astype(np.uint8)
melodies = melodies.astype(np.uint8)
harmonies = harmonies.astype(np.uint8)

input('save >>>   ')

datas = {
    'name': name,
    'type': tipo,
    'S': S,
    'X': X,
    'melodies': melodies,
    'harmonies': harmonies,
    'description': f"""JSBChorales modified {tipo} dataset: {name}. Every Tensor is a multihot reppresentation.

The chorales have been transposed from -6 to + 6 semitones, and every different transposition is present.
The dataset has been transcribed in the circle of {S}th.
X is the original transposed dataset with every transposition and no other manipulation a part from the multihot encoding.
the "melodies" array is an array of the highest voice of the chorales, while the "harmonies" array is an array of the lower voices
"""
}

joblib.dump(datas, f'datas/jsb1-{name}-{tipo}.pkl')
