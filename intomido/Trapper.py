import random as rd
from composers import *

piano = Pianoroll(16, 16)

progression = Progressions.moddy.waltz()
piano._add_group(progression*2)
mask = piano.get_blank_mask()
#mask.realtive_mask([i for i in range(0, 128, 8)], 'positive')
piano.mask(mask)
piano.plot()
piano.save_to('piano2.mid')

