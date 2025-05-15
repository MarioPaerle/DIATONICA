import joblib
import numpy as np

from composers import *

generated = joblib.load("sample4.pkl").numpy().astype(np.float16)
generated = np.round(generated).astype(np.uint8)
piano = Pianoroll(16, 8)
print(piano.grid.shape)
piano.grid[:, :100] += generated
piano.plot()

CNotes = NoteList(Chords.VImin.notes_values())
Fnotes = NoteList(Chords.IImin.notes_values())
Fmnotes = NoteList(Chords.IIImaj.notes_values())

piano.cast_to(CNotes, indicies=slice(0, 32))
piano.cast_to(Fnotes, indicies=slice(32, 64))
piano.cast_to(Fmnotes, indicies=slice(64, 96))
piano.cast_to(CNotes, indicies=slice(96, 128))
piano.plot()
piano.save_to("try1.mid")