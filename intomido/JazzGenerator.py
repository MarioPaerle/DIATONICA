import random as rd
from composers import *

progression = Progressions.w3
piano = Pianoroll(12, 16)
piano._add_group(progression)
piano.plot()
piano.save_to('piano2.mid')
