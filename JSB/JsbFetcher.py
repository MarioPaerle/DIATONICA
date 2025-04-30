import numpy as np
import json
from intomido.functions import *
import matplotlib.pyplot as plt
import joblib
S = 7  # coprimo con 12

with open("datas/chorale1.json", 'r') as f:
    chorale1 = json.load(f)

tipo = 'train'
name = 'li_5th'



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
transposed = mod_to_midi_representation(transposed, 1)
transposed = np.array(transposed)

rolled = roll(transposed, 33, 2).transpose(0, 2, 1, 3)
X_rolled = rolled[:, :-1, :, :]
Y_rolled = rolled[:, 1:, :, :]

holed = transposed
holed08 = add_random_holes(holed, p=0.8).astype(np.uint8)
holed05 = add_random_holes(holed, p=0.5).astype(np.uint8)
holed02 = add_random_holes(holed, p=0.2).astype(np.uint8)
X = transposed.astype(np.uint8)
X_rolled = X_rolled.astype(np.uint8)
Y_rolled = Y_rolled.astype(np.uint8)
plt.imshow(holed[0])
plt.show()

print(X.shape)
input('save >>>   ')

datas = {
    # 'X_rolled': X_rolled,
    # 'Y_rolled': Y_rolled,
    'name': name,
    'type': tipo,
    'S': S,
    'X': X,
    '0.5_holed': holed05,
    '0.2_holed': holed02,
    '0.8_holed': holed08,
    'description': f"""JSBChorales modified {tipo} dataset: {name}. Every Tensor is a multihot reppresentation.

The chorales have been transposed from -6 to + 6 semitones, and every different transposition is present.
The dataset has been transcribed in the circle of {S}th.
Rolled datas are sequential rolling windows tensors, X, Y are made to be used in seq2seq
Rolleds are in the shape (N_samples, 67 windows, 128 (pitches), 33 (window size)),
X is the original transposed dataset with every transposition and no other manipulation a part from the multihot encoding.
The various holed datasets are just X with random masking applied with probabilities respectively of 0.2, 0.5 and 0.8 applied.
X in the dataset is to be intended also as targets for de-holing purposes. 
"""
}

joblib.dump(datas, f'datas/jsb1-{name}-{tipo}.pkl')
