import numpy as np
import mido
import pretty_midi
from functions import multi_hot_to_midi
import matplotlib.pyplot as plt

import copy

class CopyArr(type):
    def __getattribute__(cls, name):
        attr = super().__getattribute__(name)
        return copy.deepcopy(attr)

class Note:
    def __init__(self, note, start, end, velocity=100, notation=''):
        self.note = note
        self.velocity = velocity
        self.notation = notation
        self.start = start
        self.end = end

    def transpose(self, k):
        self.note += int(k)
        return self

    def half(self):
        self.end /= 2
        self.start /= 2
        return self

    def double(self):
        self.end *= 2
        self.start *= 2
        return self

    def multiply(self, k):
        self.end *= k
        self.start *= k
        self.start, self.end = round(self.start), round(self.end)
        return self

    def move(self, k):
        self.end += k
        self.start += k
        return self

    def copy(self):
        return Note(self.note, self.start, self.end, self.velocity, self.notation)

    def __add__(self, other):
        self.note += other
        return self

    def __repr__(self):
        return f"Note({self.note}, s:{self.start}, e:{self.end},v: {self.velocity}, N:{self.notation})"

    def __str__(self):
        return f"Note({self.note}, s:{self.start}, e:{self.end},v: {self.velocity}, N:{self.notation})"

class Group:
    def __init__(self, notes):
        self.notes = notes
        self.end_pause = 0

    def transpose(self, k):
        self.notes = [n + k for n in self.notes]
        return self

    def move(self, k):
        self.notes = [n.move(k) for n in self.notes]
        return self

    def half(self):
        self.notes = [n.half() for n in self.notes]
        return self

    def double(self):
        self.notes = [n.double() for n in self.notes]
        return self

    def multiply(self, k):
        self.notes = [n.multiply(k) for n in self.notes]
        return self

    def join(self, other):
        self.notes.extend(other.notes)
        return self

    def end(self):
        return max([e.end for e in self.notes]) + self.end_pause

    def start(self):
        return min([e.start for e in self.notes])

    def duration(self):
        return self.end() - self.start()

    def copy(self):
        return Group([n.copy() for n in self.notes])

    def add_pause(self, k):
        self.end_pause += k

    def __add__(self, other):
        self.join(other.move(self.duration() - other.start()))
        return self

    def __mul__(self, k):
        for i in range(k):
            self + self.copy()
        return self

    def __repr__(self):
        return f"Group({self.notes})"

    def __str__(self):
        return f"Group({self.notes})"

class Chord(Group):
    def __init__(self, notes):
        """At creation of a Chord object the tonic is assumed to be the first note"""
        super().__init__(notes)
        self.tonic = notes[0]

    def _rivolt(self):
        self.notes = self.notes[1:].append(self.notes[0]+12)

    def get_bass(self, octave_down=1):
        return self.tonic + -12*octave_down

    def waltz(self):
        bass = self.get_bass(octave_down=1)
        duration = self.duration()
        k1 = [n.copy() for n in self.notes]
        for k in k1:
            k.multiply(1/3)
            k.move(duration//3)
            k.end -= 4

        k2 = [n.copy() for n in self.notes]
        for k in k2:
            k.multiply(1/3)
            k.move(2 * (duration//3))
        print(k2)
        print(k1)
        self.notes = k1 + k2
        self.notes.append(bass)
        return self


class Mask:
    def __init__(self, bars=16, barlen=16):
        self.lenght = bars*barlen
        self.barlen = barlen
        self.bars = bars
        self.mask = np.ones((128, self.lenght), dtype=np.uint8)

    def mask_row(self, row):
        self.mask[row] = 0

    def mask_col(self, col):
        self.mask[:, col] = 0

    def realtive_mask(self, relative_positions, mode='positive', mlt=4):
        if mode == 'positive':
            self.mask*=0
            for i in range(self.lenght):
                if i % (self.barlen*mlt) in relative_positions:
                    self.mask[:, i] = 1
        else:
            for i in range(self.lenght):
                if i % (self.barlen*mlt) in relative_positions:
                    self.mask[:, i] = 0

    def plot(self):
        plt.imshow(self.mask)
        plt.show()

class Pianoroll:
    def __init__(self, bars=16, subdivision=16):
        self.added_notes = []
        self.bars = bars
        self.subdivision = subdivision
        self.grid = np.zeros((128, self.bars*self.subdivision), dtype=np.uint8)

    def save_to(self, filename):
        multi_hot_to_midi(self.grid.T, time_per_step=.5/16).write(filename)

    def _add_note(self, note: Note):
        self.added_notes.append(note)
        start = note.start
        end = note.end
        velocity = note.velocity
        self.grid[note.note, start:end] = velocity

    def _add_group(self, group: Group):
        for note in group.notes:
            self._add_note(note)

    def add_note(self, pitch, start, end, velocity=100):
        note = Note(pitch, start, end, velocity)
        self._add_note(note)

    def transpose(self, k):
        self.grid = np.roll(self.grid, k, axis=0)

    def plot(self):
        plt.imshow(self.grid[::-1, :])
        plt.show()

    def get_blank_mask(self):
        return Mask(bars=self.bars, barlen=self.subdivision)

    def mask(self, mask):
        self.grid *= mask.mask
        return self


def chord(pitches, start, end):
    """Tonic must be the first pitch"""
    notes = [Note(p, s, e) for p, s, e in zip(pitches, start, end)]
    return Chord(notes)

def easychord(tonic, mod, start, end):
    notes = [Note(tonic, start*16, end*16) + k for k in MOD[mod]]
    return Chord(notes)

MOD = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    'maj7': [0, 4, 7, 10],
    'maj7+': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'min7+': [0, 3, 7, 11],
    'sus1': [0, 2, 7],
    'sus2': [0, 5, 7],
}

class Notes(metaclass=CopyArr):
    C =  Note(60, 0, 1, 100)
    Db = Note(61, 0, 1, 100)
    D =  Note(62, 0, 1, 100)
    Eb = Note(63, 0, 1, 100)
    E =  Note(64, 0, 1, 100)
    F =  Note(65, 0, 1, 100)
    Gb = Note(66, 0, 1, 100)
    G =  Note(67, 0, 1, 100)
    Ab = Note(68, 0, 1, 100)
    A =  Note(69, 0, 1, 100)
    Bb = Note(70, 0, 1, 100)
    B =  Note(71, 0, 1, 100)

class Chords(metaclass=CopyArr):
    Imaj =    easychord(tonic=Notes.C.note, mod='maj', start=0, end=1)
    Imin =    easychord(tonic=Notes.C.note, mod='min', start=0, end=1)
    bIImaj =  easychord(tonic=Notes.Db.note, mod='maj', start=0, end=1)
    bIImin =  easychord(tonic=Notes.Db.note, mod='min', start=0, end=1)
    IImaj =   easychord(tonic=Notes.D.note, mod='maj', start=0, end=1)
    IImin =   easychord(tonic=Notes.D.note, mod='min', start=0, end=1)
    bIIImaj = easychord(tonic=Notes.Eb.note, mod='maj', start=0, end=1)
    bIIImin = easychord(tonic=Notes.Eb.note, mod='min', start=0, end=1)
    IIImaj =  easychord(tonic=Notes.E.note, mod='maj', start=0, end=1)
    IIImin =  easychord(tonic=Notes.E.note, mod='min', start=0, end=1)
    IVmaj =   easychord(tonic=Notes.F.note, mod='maj', start=0, end=1)
    IVmin =   easychord(tonic=Notes.F.note, mod='min', start=0, end=1)
    bVmaj =   easychord(tonic=Notes.Gb.note, mod='maj', start=0, end=1)
    bVmin =   easychord(tonic=Notes.Gb.note, mod='min', start=0, end=1)
    Vmaj =    easychord(tonic=Notes.G.note, mod='maj', start=0, end=1)
    Vmin =    easychord(tonic=Notes.G.note, mod='min', start=0, end=1)
    bVImaj =  easychord(tonic=Notes.Ab.note, mod='maj', start=0, end=1)
    bVImin =  easychord(tonic=Notes.Ab.note, mod='min', start=0, end=1)
    VImaj =   easychord(tonic=Notes.A.note, mod='maj', start=0, end=1)
    VImin =   easychord(tonic=Notes.A.note, mod='min', start=0, end=1)
    bVIImaj = easychord(tonic=Notes.Bb.note, mod='maj', start=0, end=1)
    bVIImin = easychord(tonic=Notes.Bb.note, mod='min', start=0, end=1)
    VIImaj =  easychord(tonic=Notes.B.note, mod='maj', start=0, end=1)
    VIImin =  easychord(tonic=Notes.B.note, mod='min', start=0, end=1)

class Progressions(metaclass=CopyArr):
    moddy = (Chords.VImin + Chords.IImin + Chords.IIImin + Chords.VImin)
    moddy2 = Chords.VImaj.multiply(4).waltz()

if __name__ == "__main__":
    piano = Pianoroll(subdivision=16)
    """mask = piano.get_blank_mask()
    mask.realtive_mask((0, 16, 32), mode='negative')
    piano.mask(mask)"""
    prog = Progressions.moddy2
    piano._add_group(prog)
    piano.plot()
    piano.save_to("piano.mid")

