import random as rd
from composers import *

piano = Pianoroll(16, 16)

piano.add_rythmic_pattern_list([120 if i % 16 == 0 else 0 for i in range(0, 256)])
piano.add_rythmic_pattern_list([65 if i % 16 == 4 else 0 for i in range(0, 256)])
piano.add_rythmic_pattern_list([40 if i % 16 == 8 else 0 for i in range(0, 256)])
piano.add_rythmic_pattern_list([65 if i % 16 == 12 else 0 for i in range(0, 256)])

piano.plot()
piano.save_to('hh2.mid')