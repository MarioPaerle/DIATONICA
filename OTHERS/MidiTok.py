from miditok import REMI, MIDITokenizerConfig
from pathlib import Path

# Define the tokenizer
config = MIDITokenizerConfig()
tokenizer = REMI(config)

# Folder with your MIDI files
midi_path = Path("./midi_dataset")
tokenized_path = Path("./tokenized_dataset")
tokenized_path.mkdir(parents=True, exist_ok=True)

# Tokenize and save
tokenizer.tokenize_midi_dataset(midi_path, tokenized_path, save_as_json=True)
