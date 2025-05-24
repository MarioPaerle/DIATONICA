from intomido.composers import *
import intomido.functions as f


piano = Pianoroll()

progressions = [
    (Chords.VImin + Chords.IImin + Chords.IIImaj + Chords.VImin).to_chord_progression().enlarge().multiply(16),
]

piano._add_group(progressions[0])
piano.plot()
piano.save_to('s.mid')