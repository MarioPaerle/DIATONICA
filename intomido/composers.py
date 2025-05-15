import numpy as np
import random as rd

from functions import multi_hot_to_midi, nearest, cast_pianoroll_to_scale
import matplotlib.pyplot as plt
from OTHERS.DiscretePolynomialApproximators import polyntepolate, melodic_interpolate

import copy
import warnings

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
        return f"Note({self.note}, s:{self.start}, e:{self.end})"

    def __str__(self):
        return f"Note({self.note}, s:{self.start}, e:{self.end},v: {self.velocity}, N:{self.notation})"

class Pause:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.velocity = 0
        self.note = 0

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

    def swing(self, swing_amount=1):
        for note in self.notes:
            if note.start % 8 != 0:
                note.start += swing_amount
                note.end += swing_amount

        return self

    def notes_values(self):
        return [n.note for n in self.notes]

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

    def __len__(self):
        return len(self.notes)

class Pattern:
    def __init__(self, notes, start, subdivision):
        self.pattern = notes
        self.subdivision = subdivision
        self.start = start

    def transpose(self, k):
        for note in self.pattern:
            if isinstance(note, int):
                note += k

class Chord(Group):
    def __init__(self, notes):
        """At creation of a Chord object the tonic is assumed to be the first note"""
        super().__init__(notes)
        self.tonic = notes[0]
        self.separed_chords = []

    def get_separed_chords(self):
        return self.separed_chords

    def _rivolt(self):
        self.notes = self.notes[1:].append(self.notes[0]+12)

    def get_bass(self, octave_down=1):
        return self.tonic.copy() + -12*octave_down

    def __add__(self, other):
        super().__add__(other)
        if isinstance(other, Chord):
            if len(self.get_separed_chords()) == 0:
                self.separed_chords.append(self.copy())
            else:
                self.separed_chords.append(other.copy())
        else:
            warnings.warn("If you're trying to generate a multi chord progression, you should add a Chord not a Group!")
        return self

    def waltz(self, endcut=3, bassadd=0):
        # TODO: Waltzer fixes, the chord does not waltz correctly!
        if self.start() != 0:
            pass
            #raise Exception("Chord.waltz() is not implemented for chords that start differently from 0")
        if len(self.get_separed_chords()) == 0:
            rstart = 0 # self.start()
            _ = self.move(-rstart)
            bass = self.get_bass(octave_down=1) + bassadd
            duration = self.duration()
            k1 = [n.copy() for n in self.notes]
            for k in k1:
                k.multiply(1/3)
                k.move(duration//3)
                k.end -= duration//(4*endcut)
                k.start += 0.5

            k2 = [n.copy() for n in self.notes]
            for k in k2:
                k.multiply(1/3)
                k.move(2 * (duration//3) + 0.5)

            self.notes = k1 + k2
            self.notes.append(bass)
            self.move(rstart)
            return self
        else:
            raise Exception("you're probably trying to waltz a Chord object with more than a group! use .to_chord_progression().waltz() instead")

    def get_pitches(self, idx, transpose=12):
        return [n.note + transpose for n in self.separed_chords[idx].notes]

    def copy(self):
        group = super().copy()
        c = Chord(group.notes)
        c.separed_chords = self.separed_chords.copy()
        return c

    def __str__(self):
        return f"Chord({self.notes})"

    def __repr__(self):
        return f"Chord({self.notes})"

    def to_chord_progression(self):
        return ChordsProgression(self.separed_chords)

class ChordsProgression:
    def __init__(self, chords: list[Chord]):
        assert type(all([type(c) == Chord for c in chords]))
        self.chords = chords
        print(self.chords)

    def transpose(self, k):
        for chord in self.chords:
            chord.transpose(k)

    def waltz(self, e=3):
        for chord in self.chords:
            chord.waltz(e)
        return self

    def to_chord(self):
        chord_ = self.chords[0]
        for chord in self.chords[1:]:
            chord_ += chord
        return chord_

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ChordsProgression(self.chords[idx])
        else:
            return self.chords[idx]

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

class NoteList:
    """A NoteList holds all octave-equivalents of given root pitches in the MIDI range 0–128."""

    def __init__(self, roots: list[int]):
        self.roots = roots
        self.notes = self._generate_notes()

    def _generate_notes(self) -> list[int]:
        """Generate all notes for each root by adding multiples of 12 (octaves)
        so long as the resulting MIDI value is between 0 and 128 inclusive."""
        notes = set()
        for root in self.roots:
            octave = -((root) // 12)
            while True:
                pitch = root + octave * 12
                if pitch > 128:
                    break
                if 0 <= pitch <= 128:
                    notes.add(pitch)
                octave += 1
        return sorted(notes)

    def add_note(self, pitch):
        self.notes.append(pitch)
        return self

    def __add__(self, semitones: int) -> "NoteList":
        """Return a new NoteList with all notes shifted up by `semitones`."""
        shifted = NoteList(self.roots)
        shifted.notes = [n + semitones for n in self.notes]
        return shifted

    def __sub__(self, semitones: int) -> "NoteList":
        """Return a new NoteList with all notes shifted down by `semitones`."""
        return self + (-semitones)

    def __getitem__(self, idx):
        """Allow indexing directly into the generated notes."""
        return self.notes[idx]

    def __len__(self):
        """Number of notes in the list."""
        return len(self.notes)

    def list(self) -> list[int]:
        """Get the underlying list of MIDI note numbers."""
        return self.notes.copy()

    def __repr__(self):
        return f"NoteList({self.roots})"

    def __str__(self):
        return f"NoteList({self.roots}):: {self.notes}"

class Pianoroll:
    def __init__(self, bars=16, subdivision=16):
        self.added_notes = []
        self.bars = bars
        self.subdivision = subdivision
        self.grid = np.zeros((128, self.bars*self.subdivision), dtype=np.uint8)

    def save_to(self, filename):
        multi_hot_to_midi(self.grid.T, time_per_step=.5/self.subdivision).write(filename)

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

    def add_list_pattern(self, pattern, subdivision=4, start=0, clamp_end=float('inf'), transpose=0):
        """pattern must be like [67, 65, '-', ...]"""
        time = start
        last_note = None
        for pitch in pattern:
            if pitch not in ('_', '-'):
                pitch += transpose
                note = Note(pitch, time, min(time+subdivision, clamp_end), 100)
            elif pitch == '-':
                note = last_note
                last_note.end += subdivision
            else:
                note = Pause(time, min(time+subdivision, clamp_end))

            last_note = note
            self._add_note(note)
            time = min(time+subdivision, clamp_end)
        return pattern

    def add_listed_pattern(self, p, start=0, clamp_end=float('inf')):
        """pattern must be like [67, 65, '-', ...]
        Differences with add_list_pattern is that here the subdivision is also passed in the p"""
        time = start
        pattern, subdivision = p
        last_note = None
        for pitch in pattern:
            if pitch not in ('_', '-'):
                pitch += 0
                note = Note(pitch, time, min(time+subdivision, clamp_end), 100)
            elif pitch == '-':
                note = last_note
                last_note.end += subdivision
            else:
                note = Pause(time, min(time+subdivision, clamp_end))

            last_note = note
            self._add_note(note)
            time = min(time+subdivision, clamp_end)

    def add_rythmic_pattern_list(self, pattern_velocities_list: list, note=72):
        self.grid[note, :] += np.array(pattern_velocities_list, dtype=np.uint8)

    def cast_to(self, scale, indicies=None):
        if indicies is None:
            self.grid = cast_pianoroll_to_scale(self.grid.T, scale).T
        else:
            self.grid[:, indicies] = cast_pianoroll_to_scale(self.grid[:, indicies].T, scale).T

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
    'minj1': [0, 2, 3, 7],
    'minj2': [2, 3, 7],
    'maj7': [0, 4, 7, 10],
    'maj7+': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'min7+': [0, 3, 7, 11],
    'sus1': [0, 2, 7],
    'sus2': [0, 5, 7],
    'nap': [0, 3, 8]
}

PATTERNS = {
    '5_to_6_1': (['_', 71, 76, 80, 84, 83], 11),
    '5_to_6_2': (['_', '_', '_', '_', '_', '_', 84, 83], 8),
    '5_to_6_3': (['_', '_', '_', '_', '_', 86, 84, 83], 8),
    '5_to_6_4': (['_', '_', '_', '_', '_', 88, 84, 83], 8),
    '5_to_6_5': ([76, '_', '_', '_', '_', 76, 78, 80], 8),
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

    Napulitan = easychord(tonic=Notes.B.note, mod='min', start=0, end=1)
class Progressions(metaclass=CopyArr):
    moddy = (Chords.VImin + Chords.IImin + Chords.IIImin + Chords.VImin)
    moddy2 = (Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(4).transpose(-12)*2
    w1 = (Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(4)*2
    w2 = (Chords.VImin.waltz() + Chords.IImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(4)*2
    w3 = (Chords.IImin.waltz() + Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz()).multiply(3)*2

    # Real Waltzers
    op64n2_a = (Chords.VImin.waltz() + Chords.IIImaj.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz())# .multiply(3)
    op64n2_b = (Chords.IVmaj.waltz() + Chords.IVmaj.waltz() + Chords.IIImaj.waltz() + Chords.VImin.waltz())# .multiply(3)
    op64n2 = op64n2_a.copy() + op64n2_b.copy()
class Scales:
    Cmajor = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108, 110, 112, 113, 115, 117, 119, 120, 122, 124, 125, 127]
    Cmajor

if __name__ == "__main__":
    Progs = [Progressions.op64n2]
    """This demo generates a basic Waltzer"""
    piano = Pianoroll(subdivision=12, bars=32)
    prog = rd.choice(Progs)
    piano._add_group(prog)
    piano.save_to('output.mid')
    quit()
    piano.add_listed_pattern(
        PATTERNS[f'5_to_6_{rd.randint(1, 5)}'], start=32*4, clamp_end=48*4
    )
    pattern = piano.add_list_pattern(
        melodic_interpolate(
            [0, 8, 16],
            [rd.choice(prog.get_pitches(0)), rd.choice(prog.get_pitches(0)), rd.choice(prog.get_pitches(1))],
            24,
            16,
            scale=prog.get_pitches(0)),
        subdivision=4
    )
    piano.add_list_pattern(
        pattern,
        subdivision=4,
        transpose=2,
        start=64
    )
    piano.add_list_pattern(
        pattern,
        subdivision=4,
        transpose=0,
        start=64*4
    )
    piano.add_list_pattern(
        pattern,
        subdivision=4,
        transpose=2,
        start=64*5
    )
    pattern = piano.add_list_pattern(
        melodic_interpolate(
            [0, 16, 32, 64],
            [rd.choice(prog.get_pitches(2)), rd.choice(prog.get_pitches(2)), rd.choice(prog.get_pitches(3)), rd.choice(prog.get_pitches(3))],
            128,
            64,
            scale=prog.get_pitches(3)),
        start=64*6,
        subdivision=4
    )


    piano.add_note(81, 48*4, 64*4)
    piano.plot()
    piano.save_to("piano.mid")