from miditok import CompoundWord
from symusic import Score

# Configura tokenizer per includere note multiple e chord
tokenizer = CompoundWord(
    encoding_form="compound",
    use_chords=True,
    beat_size=4,             # 4 step per battuta
    tokens_map={
        "Pitch": list(range(128)),
        "Duration": [1,2,4,8],
        "Velocity": list(range(1, 128, 8))
    }
)

# Tokenizza in finestre di 4 step (quarter notes)
tokens = tokenizer.tokenize(path_to_midi)
print(tokens)