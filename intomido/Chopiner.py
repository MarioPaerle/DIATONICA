from intomido.composers import  *
import random as rd

"""midi = midi_to_numpy("3foturhSportswear44.mid").astype(np.uint8)
midi = midi[:, 0::3]"""
"""plt.imshow(midi)
plt.show()"""

possible_progressions = [
    "Am Dm E Am | F Dm E Am",
    "Am E E Am | F Dm E Am",
    "Am Am E Am | Dm Dm6 E Am",
]

phrases = [
    "Mel Mel Close | Mel Mel ScaleClose",
    "Note Note ScaleClose | Note Scale CascadeClose",
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
progression = rd.choice(possible_progressions)
phrase = rd.choice(phrases)

if True:
    closing_note = ['tonic', 'third', 'fifth', 'second']
    SCALE_CLOSES = {
        'tonic': [
            [62, 63, 64, 65, 66, 67],
            [76, 64, 74, 68, 72, 71],
            [75, 74, 73, 72, 71, 70],
            [59, 60, 61, 62, 68, 64]
        ]
    }

print(f"Selected Progression  {rd.choice(possible_progressions)}")
print(f"Selected phrases  {rd.choice(phrases)}")
print(f"Selected hows  {rd.choices(melhows, k=8)}")


