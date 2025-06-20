from intomido.encodings import StringBPE
from GridTokenizer2 import tokenize, add_bars, transpose
import os
import joblib

files = os.listdir("MuseScoreMIDIS2")
midis = []
entokened = []
if True:
    for file in files:
        if file.endswith(".mid"):
            try:
                io = tokenize(f'MuseScoreMIDIS2/{file}', cycle_length_beats=2, subdivisions_per_beat=4)
                io = add_bars(io)
                for i in range(0, 13):
                    io_scaled = transpose(io, i)
                    entokened.append(io_scaled)
            except Exception as e:
                print(file, e)

    print(sum([len(k) for k in entokened]))


ENCODER = StringBPE()
ENCODER.train(entokened, num_merges=100)

print(len(ENCODER.vocab))
print(max(ENCODER.token_to_id.values()))
print(min(ENCODER.token_to_id.values()))
print(ENCODER.token_to_id)
input()

tokenized = []
for file in files:
    if file.endswith(".mid"):
        try:
            io = tokenize(f'MuseScoreMIDIS2/{file}', cycle_length_beats=2, subdivisions_per_beat=4)
            io = add_bars(io)
            for i in range(0, 13):
                io_scaled = transpose(io, i)
                numerical = ENCODER.tokenize_vector(io_scaled)
                tokenized.append(ENCODER.tokenize_vector(io_scaled))
        except Exception as e:
            print(file, e)

joblib.dump(tokenized, "tokenized_test7.pkl")
joblib.dump(ENCODER, "tokenizer_test7.pkl")

