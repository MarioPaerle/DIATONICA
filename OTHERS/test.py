import pandas as pd
from tokenizers import convert_to_midi
import numpy as np

splits = {'train': 'data/train-00000-of-00001-4ac3ace894e8a8c0.parquet', 'validation': 'data/validation-00000-of-00001-1ed2f10cc97689f1.parquet', 'test': 'data/test-00000-of-00001-7df6e8020a9c62b5.parquet'}
df = pd.read_parquet("hf://datasets/JasiekKaczmarczyk/giant-midi-quantized/" + splits["train"])

p = df.iloc[10000]
print(p)
midi_file = convert_to_midi(p)

output_filename = "converted_output.mid"
midi_file.save(output_filename)
print(f"MIDI file saved as {output_filename}")