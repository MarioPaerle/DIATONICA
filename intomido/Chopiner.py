from intomido.composers import  *
import random as rd

"""midi = midi_to_numpy("3foturhSportswear44.mid").astype(np.uint8)
midi = midi[:, 0::3]"""
"""plt.imshow(midi)
plt.show()"""

RESOLUTION = 12

possible_progressions = {
    "Am Dm E Am | F Dm E Am": (Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IVmaj.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION),
    "Am E E Am | F Dm E Am": (Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IVmaj.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION),
    "Am Am E Am | Dm Dm6 E Am" : (Chords.VImin.waltz() + Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION),
}

phrases = [
    "Mel Mel Close | Mel Mel ScaleClose",
    "Mel Mel Mel Mel | Mel Mel ScaleClose",
    "Mel Mel Mel Mel | Mel Mel Close",
    "Mel Mel Mel Mel | Mel Mel CascadeClose",
    "Note Note ScaleClose | Note Scale CascadeClose",
    "Note Note MelClose | Note Note Close",
    "Note Note MelClose | Note Note Close",
]

melhows = [
    "Simple Repeat",
    "Casted Repeat",
    "UpCast Repeat",
    "DownCast Repeat",
]

closehows = [
    "Ballerin",

]
if True:
    closing_note = ['tonic', 'third', 'fifth', 'second']
    SCALE_CLOSES = {
        'tonic': [
            [62, 63, 64, 65, 66, 67],
            [75, 74, 73, 72, 71, 70],
            [59, 60, 61, 62, 68, 64],
            [69, 70, 71, 72, 73, 68],

        ]
    }

    CLOSE = {
        'tonic':[
        [76, 64, 74, 68, 72, 71],
        [64, 72, 62, 71, 59, 68],
        ]
    }

    MEL_PATTERNS = {
        'tonic': [
            [76, 69, 72, '-', '-', '-'],
            [69, 71, 72, 64, 76, '-'],
        ]
    }

progression = rd.choice(list(possible_progressions.keys()))
phrase = rd.choice(phrases)
melhow = rd.choices(melhows, k=8)

print(f"Selected Progression  {progression}")
print(f"Selected phrases  {phrase}")
print(f"Selected hows  {melhow}")

roll = Pianoroll(9, RESOLUTION)
prog = possible_progressions[progression]
roll._add_group(prog)
roll.plot()

barcount = 0
patterns = []
thispattern = []
chords = prog.get_separed_chords()


for i, word in enumerate(phrase.split()):
    cast = NoteList(list(set(chords[i].copy().notes_values())))
    if word == '|':
        barcount += 1

    elif word == 'Note':
        thispattern = ['-'] * RESOLUTION
        thispattern[0] = rd.choice(chords[i].copy().notes_values()) + 12
        patterns.append(thispattern)
        roll.add_list_pattern(thispattern, 1, start=i*RESOLUTION, pedal_on=False)

    elif word in ('Close', 'MelClose'):
        thispattern = rd.choice(CLOSE['tonic'])
        mlt = RESOLUTION // 6
        roll.add_list_pattern(thispattern, mlt, start=i*RESOLUTION, transpose=12)
        thispattern = ['-'] * RESOLUTION
        thispattern[0] = 81
        roll.add_list_pattern(thispattern, 1, start=(i+1)*RESOLUTION)

    elif word in ('Mel'):
        thispattern = rd.choice(MEL_PATTERNS['tonic'])
        mlt = RESOLUTION // 6
        roll.add_list_pattern(thispattern, mlt, start=i*RESOLUTION, transpose=12)
        thispattern = ['-'] * RESOLUTION
        thispattern[0] = 81
        roll.add_list_pattern(thispattern, 1, start=(i+1)*RESOLUTION)

    elif word in ('ScaleClose', 'CascadeClose'):
        thispattern = rd.choice(SCALE_CLOSES['tonic'])
        mlt = RESOLUTION // 6
        roll.add_list_pattern(thispattern, mlt, start=i*RESOLUTION, transpose=12)
        thispattern = ['-'] * RESOLUTION
        thispattern[0] = 81
        roll.add_list_pattern(thispattern, 1, start=(i+1)*RESOLUTION)

    # roll.plot()
roll.save_to('roll.mid')
roll.plot()
roll.play()

