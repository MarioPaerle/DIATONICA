import random as rd
from composers import *

piano = Pianoroll(16, 16)

"""progression = Progressions.moddy.to_chord_progression().waltz().to_chord()
piano._add_group(progression)
mask = piano.get_blank_mask()
#mask.realtive_mask([i for i in range(0, 128, 8)], 'positive')
piano.mask(mask)
piano.plot()
piano.save_to('piano2.mid')
"""
# piano.add_rythmic_pattern_list([120 if i % 8 == 0 else 0 for i in range(0, 256)])
piano.add_rythmic_pattern_list([80 if i % 4 == 0 else 0 for i in range(0, 256)])
piano.plot()
piano.save_to('hh2.mid')