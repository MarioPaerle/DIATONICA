from intomido.encodings import StringBPE
from tokenizers import EasyTok
import os
from mido import MidiFile

files = os.listdir("MuseScoreMIDIS")
midis = []
entokened = []
for file in files:
    try:
        midis.append(file)
        midi = MidiFile(f"MuseScoreMIDIS/{file}")
        tok = EasyTok(midi)
        tok.tokenize(transpose=False)
        for token in tok.tolist():
            entokened.append(token)
    except:
        print(f"Error with {file}")


ENCODER = StringBPE()
ENCODER.train(entokened, num_merges=100)
#print(ENCODER.vocab)
print(len(ENCODER.vocab))

