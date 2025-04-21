from intomido.encodings import StringBPE
from GridTokenizer import tokenize, add_bars
import os
import GridTokenizer
import joblib

files = os.listdir("MuseScoreMIDIS2")
midis = []
entokened = []
for file in files:
    if file.endswith(".mid"):
        try:
            io = tokenize(f'MuseScoreMIDIS2/{file}', cycle_length_beats=4, subdivisions_per_beat=8)
            io = add_bars(io)
            entokened.append(io)
        except:
            print(file)
    """try:
        midis.append(file)
        midi = MidiFile(f"MuseScoreMIDIS/{file}")
        tok = EasyTok(midi)
        tok.tokenize(transpose=False)
        for token in tok.tolist():
            entokened.append(token)
    except:
        print(f"Error with {file}")"""

print(sum([len(k) for k in entokened]))


ENCODER = StringBPE()
ENCODER.train(entokened, num_merges=500)

tokenized = []
for file in files:
    if file.endswith(".mid"):
        try:
            io = tokenize(f'MuseScoreMIDIS2/{file}')
            io = add_bars(io)
            tokenized.append(ENCODER.tokenize_vector(io))
        except:
            print(file)

joblib.dump(tokenized, "tokenized_test2.pkl")
joblib.dump(ENCODER, "tokenizer_test2.pkl")

