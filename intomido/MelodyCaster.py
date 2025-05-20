import random as rd
import numpy as np
from composers import *
from intomido.functions import midi_to_numpy

"""midi = midi_to_numpy("3foturhSportswear44.mid").astype(np.uint8)
midi = midi[:, 0::3]"""
"""plt.imshow(midi)
plt.show()"""

progression = Progressions.op64n2.multiply(12*3).transpose(-12)
piano = Pianoroll(subdivision=16, bars=32)


I = NoteList(Chords.VImin.notes_values())
II = NoteList(Chords.IIImaj.notes_values()).add_note(69).add_note(65)
III = NoteList(Chords.IIImaj.notes_values()).add_note(69).add_note(65)
IV = NoteList(Chords.VImin.notes_values())

piano._add_group(progression)
piano.add_list_pattern([64, '-', 69, 72, 65], steps=12)
piano.add_list_pattern([65, '-', 72, 69, 64], steps=12, start=16 * 6)


piano.cast_to(I, indicies=slice(0, 48))
piano.cast_to(II, indicies=slice(48, 48*2))
piano.cast_to(III, indicies=slice(48*2, 48*3))
piano.cast_to(IV, indicies=slice(48*3, 48*4))

piano.add_list_pattern([64, '-', 69, 72, 65], steps=12, start=48 * 4)

pattern = piano.add_list_pattern(
        melodic_interpolate(
            [0, 8, 16],
            [rd.choice(progression.get_pitches(1)), rd.choice(progression.get_pitches(1)), rd.choice(progression.get_pitches(2))],
            24,
            16,
            scale=progression.get_pitches(0)),
        steps=6,
    start=48*6
    )

piano.cast_to(I, indicies=slice(48*4, 48*5))
piano.cast_to(II, indicies=slice(48*6, 48*7))
piano.cast_to(III, indicies=slice(48*8, 48*9))
piano.cast_to(IV, indicies=slice(48*9, 48*10))

piano.save_to('output.mid')
piano.plot()

"""progression = Progressions.op64n2.multiply(3).transpose(-24) * 2
piano = Pianoroll(subdivision=16, bars=64)

I = NoteList(Chords.VImin.notes_values())
II = NoteList(Chords.IIImaj.notes_values()).add_note(69).add_note(65)
III = NoteList(Chords.IIImaj.notes_values()).add_note(69).add_note(65)
IV = NoteList(Chords.VImin.notes_values())

# PART A
if True:
    piano._add_group(progression)
    piano.add_list_pattern([76, '-', 76, 72, '-', '-', 71, '-', '-', 71, 72, '-', '-', 64], subdivision=6)
    piano.add_list_pattern([76, '-', 76, 72, '-', '-', 71, '-', '-', 71, 72, '-', '-', 64], subdivision=6, start=16*6)


    piano.cast_to(I, indicies=slice(0, 48))
    piano.cast_to(II, indicies=slice(48, 48*2))
    piano.cast_to(III, indicies=slice(48*2, 48*3))
    piano.cast_to(IV, indicies=slice(48*3, 48*4))

# PART B
if True:
    piano.add_list_pattern([76, '-', 76, 72, '-', '-', 71, '-', '-', 71, 72, '-', '-', 64], subdivision=6, start=16*12)

    pattern = piano.add_list_pattern(
            melodic_interpolate(
                [0, 8, 16],
                [rd.choice(progression.get_pitches(1)), rd.choice(progression.get_pitches(1)), rd.choice(progression.get_pitches(2))],
                16,
                16,
                scale=progression.get_pitches(0)),
            subdivision=6,
        start=48*6
        )

    piano.cast_to(I, indicies=slice(48*4, 48*5))
    piano.cast_to(II, indicies=slice(48*6, 48*7))
    piano.cast_to(III, indicies=slice(48*8, 48*9))
    piano.cast_to(IV, indicies=slice(48*9, 48*10))

# PART C
if True:
    piano.add_list_pattern([76, '-', 76, 72, '-', '-', 71, '-', '-', 71, 72, '-', '-', 64], subdivision=6, start=16 * 24)
    piano.add_list_pattern([76, '-', 76, 72, '-', '-', 71, '-', '-', 71, 72, '-', '-', 64], subdivision=6, start=16 * 30)

    piano.add_chromatic_scale(
        76,
        64,
        start=48 * 12
    )
    piano.add_chromatic_scale(
        64,
        76,
        steps=(76 - 64) // 2,
        start=48 * 13,
        subdivision=8
    )
    piano.add_listed_pattern(
        PATTERNS[f'5_to_6_{rd.randint(1, 5)}'], start=48*13, clamp_end=48*16
    )

    piano.cast_to(I, indicies=slice(48 * 11, 48 * 12))
    piano.cast_to(II, indicies=slice(48 * 13, 48 * 14))
    piano.cast_to(III, indicies=slice(48 * 15, 48 * 16))
    piano.cast_to(IV, indicies=slice(48 * 17, 48 * 18))


piano.save_to('output_turndownforwhat2.mid')
piano.plot()"""