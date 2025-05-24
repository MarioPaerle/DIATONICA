from intomido.composers import  *
import random as rd
import streamlit as st
from intomido.composers import Chords as c


RESOLUTION = 12

possible_progressions = {
    "Am Dm E Am | F Dm E Am": (Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IVmaj.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION), # .pedal().to_chord(),
    "Am E E Am | F Dm E Am": (Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IVmaj.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION), # .pedal().to_chord(),
    "Am Am E Am | Dm Dmnap E Am" : (Chords.VImin.waltz() + Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.Napulitan.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION), # .pedal().to_chord(),
    "F Am E Am | Dm Dmnap E Am" : (Chords.IVmaj.waltz() + Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.Napulitan.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION), # .pedal().to_chord(),
    # "C F Fm C | G Dm E Am" : (Chords.Imaj.waltz() + Chords.IVmaj.waltz() + Chords.IVmin.waltz() + Chords.Imaj.waltz() + Chords.Vmaj.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(RESOLUTION), # .pedal().to_chord(),
}

phrases = [
    "Mel Mel Close | Mel Mel ScaleClose",
    "Mel Mel Mel | Mel Mel ScaleClose",
    "Mel Mel Mel | Mel Mel Close",
    "Mel Mel Mel | Mel Mel CascadeClose",
    "Note Note ScaleClose | Note Scale CascadeClose",
    "Note Note MelClose | Note Note Close",
    "Note Note MelClose | Note Note Close",
]

melhows = [
    "free",
    "free",
    "free",
    "free",
    "free",
    "free",
    "Simple Repeat",
    "Casted Repeat",
    "UpCast Repeat",
    "DownCast Repeat",
]

closehows = [
    "Ballerin",

]

if True:
    rivolts = [
        10,
        20,
        11,
        22,
    ]
    closing_note = ['tonic', 'third', 'fifth', 'second']
    SCALE_CLOSES = {
        'tonic': [
            [62, 63, 64, 65, 66, 67],
            [75, 74, 73, 72, 71, 70],
            [59, 60, 61, 62, 68, 64],
            [69, 70, 71, 72, 73, 68],
            [69, 70, 71, 72, 73, 68],
            [81, 76, 72, 71, 74, 71],
            [77, 76, 72, 71, 74, 71],
            [77, 76, 72, 69, 68, 74],

        ]
    }

    CLOSE = {
        'tonic':[
        [76, 64, 74, 68, 72, 71],
        [64, 72, 62, 71, 59, 68],
        [74, '-', '-', 76, 68, 72],
        [74, '-', 76, 68, 72, 71],
        ]
    }

    MEL_PATTERNS = {
        'tonic': [
            [76, 69, 72, '-', '-', '-'],
            [69, 71, 72, 64, 76, '-'],
            ['-', 76, '-', '-', '-', '-'],
            ['-', '-', 76, '-', '-', '-'],
            ['-', 72, '-', '-', '-', '-'],
            ['-', '-', 72, '-', '-', '-'],
            ['-', 76, 74, 72, 71, 69, 69],
            ['-', 76, 74, 72, 77, 76, 76],
        ]
    }

def chopiner():
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
    last_pattern = []
    lastm_mel = None
    chords = prog.get_separed_chords()


    for i, word in enumerate(phrase.split()):
        how = melhow[i]
        cast = NoteList(list(set(chords[i].copy().notes_values())))

        if word == '|':
            barcount += 1

        elif word == 'Note':
            if how == 'free':
                thispattern = ['-'] * RESOLUTION
                thispattern[0] = rd.choice(chords[i].copy().notes_values()) + 12
                last_pattern = thispattern
                patterns.append(thispattern)
                roll.add_list_pattern(thispattern, 1, start=i*RESOLUTION, pedal_on=False, deltastart=1)
            else:
                thispattern = cast_list(last_pattern, cast.list())
                roll.add_list_pattern(thispattern, 1, start=i * RESOLUTION, pedal_on=False, deltastart=1)

        elif word in ('Close', 'MelClose'):
            thispattern = rd.choice(CLOSE['tonic'])
            mlt = RESOLUTION // 6
            roll.add_list_pattern(thispattern, mlt, start=i*RESOLUTION, transpose=12)
            thispattern = ['-'] * RESOLUTION
            thispattern[0] = 81
            roll.add_list_pattern(thispattern, 1, start=(i+1)*RESOLUTION, clamp_end=(i+2)*RESOLUTION)

        elif word in ('Mel'):
            if how == 'free' or lastm_mel is None:
                thispattern = rd.choice(MEL_PATTERNS['tonic'])
                mlt = RESOLUTION // 6
                roll.add_list_pattern(thispattern, mlt, start=i*RESOLUTION, transpose=12)
                last_pattern = thispattern
                lastm_mel = thispattern
            else:
                st.write('casted')
                thispattern = lastm_mel
                mlt = RESOLUTION // 6
                roll.add_list_pattern(cast_list(thispattern, scale=cast), mlt, start=i * RESOLUTION, transpose=12)

        elif word in ('ScaleClose', 'CascadeClose'):
            thispattern = rd.choice(SCALE_CLOSES['tonic'])
            mlt = RESOLUTION // 6
            roll.add_list_pattern(thispattern, mlt, start=i*RESOLUTION, transpose=12)
            thispattern = ['-'] * RESOLUTION
            thispattern[0] = 81
            roll.add_list_pattern(thispattern, 1, start=(i+1)*RESOLUTION, clamp_end=(i+2)*RESOLUTION)

    roll.plot()
    roll.transpose(rd.randint(-9, 4))

    midi = roll.midi_human()
    midi.write('roll.mid')
    return roll, midi

st.title("Chopiner 🎹")
st.subheader('v0.1')
st.write("Here's a simple Waltzer generator of just 8 bars 😒")

if st.button("generate"):
    waltzer, midi = chopiner()

    st.pyplot(waltzer.plot())
    st.audio(waltzer.toaudio(), sample_rate=44_100)