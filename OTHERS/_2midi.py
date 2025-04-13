from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
import os

KEY_IMPLICI_TRANSPOSE = {
    'C': 0,
    'C#': -1,
    'Db': -1,
    'D': -2,
    'D#': -3,
    'Eb': -3,
    'E': -4,
    'F': -5,
    'F#': -6,
    'Gb': -6,
    'G': -7,
    'G#': -8,
    'Ab': -8,
    'A': -9,
    'A#': -10,
    'Bb': -10,
}

class MidiTok:
    def __init__(self, midi: MidiFile):
        self.midi = midi
        for msg in midi:
            if msg.type == 'time_signature':
                self.numerator = msg.numerator
                self.denominator = msg.denominator
                self.musetime = f"{self.numerator}/{self.denominator}"
                self.clock = msg.clocks_per_click
            if msg.type == 'key_signature':
                self.key = msg.key
        self.value = []
        self.str_value_on = []
        self.str_value_off = []
        self.notes_on =  []
        self.notes_off = []
        self.implicit_transpose = KEY_IMPLICI_TRANSPOSE[self.key]
        self.events = dict()
        self.ticker = midi.ticks_per_beat
        self.all_ticker = self.ticker * self.numerator
        self.midi_track = midi.tracks
        self.n_tracks = len(self.midi_track)
        time = 0
        for msg in midi.merged_track:
            time += msg.time if len(self.notes_on) > 0 else 0
            if msg.type == 'note_on':
                if msg.velocity == 0:
                    self.notes_off.append((time, msg))
                else:
                    self.notes_on.append((time, msg))
            elif msg.type == 'note_off':
                self.notes_off.append((time, msg))

        self.ordered_notes_on =  sorted(self.notes_on, key=lambda x: x[0])
        self.ordered_notes_off = sorted(self.notes_off, key=lambda x: x[0])

    def tokenize(self, transpose=False):
        if transpose:
            delta = self.implicit_transpose
        else:
            delta = 0
        for abs_time, note in self.ordered_notes_on:
            if abs_time in self.events:
                self.events[abs_time].append((note, 'on'))
            else:
                self.events[abs_time] = [(note, 'on')]

        for abs_time, note in self.ordered_notes_off:
            if abs_time in self.events:
                self.events[abs_time].append((note, 'off'))
            else:
                self.events[abs_time] = [(note, 'off')]

        for abs_time in self.events:
            msgs = self.events[abs_time]
            notes_on = [msg[0].note + delta for msg in msgs if msg[1] == 'on']
            notes_off = [msg[0].note + delta for msg in msgs if msg[1] == 'off']
            timeenc = (abs_time % self.all_ticker) // self.denominator
            string_on = ""
            string_off = ""
            for i, note in enumerate(notes_on):
                if i > 0:
                    string_on += "+"
                string_on += f"{note}"
            string_on += f".{timeenc}"

            for i, note in enumerate(notes_off):
                if i > 0:
                    string_off += "+"
                string_off += f"{note}"
            string_off += f".{timeenc}" if len(string_off) > 0 else ""

            self.str_value_on.append(string_on)
            if len(string_off) > 0:
                self.str_value_off.append(string_off)
        # print(self.str_value_off)

    def __str__(self):
        baseline = f"{self.clock}, {self.numerator}, {self.denominator}"
        return baseline + "\n" +  " ".join(self.str_value_on) + '\n' + " ".join(self.str_value_off)

class MidiDeTok:
    def __init__(self, midi: str):
        self.str = midi
        self.events = dict()
        self.midi = MidiFile()
        self.track = MidiTrack()
        self.midi.tracks.append(self.track)

    def parse(self):
        self.baseline, self.notes_on, self.notes_off = self.str.split('\n')
        numerator = int(self.baseline.split(', ')[1])
        denominator = int(self.baseline.split(', ')[2])
        self.track.append(MetaMessage('time_signature', numerator=numerator, denominator=denominator))

        last_time = 0
        for msgs in self.notes_on.split():
            notes, time = msgs.split('.')
            notes = notes.split('+')
            time = int(time)
            for i, note in enumerate(notes):
                if note == '':
                    continue
                self.track.append(Message('note_on', note=int(note), velocity=100, time=40 if i == 0 else 0))
                self.track.append(Message('note_off', note=int(note), velocity=0, time=0 if i == 0 else 100))
            last_time = time
        """for msgs in self.notes_off.split():
            notes, time = msgs.split('.')
            notes = notes.split('+')
            time = int(time)
            for note in notes:
                if note == '':
                    continue
                self.track.append(Message('note_off', note=int(note), velocity=0, time=time))"""

    def export(self, filename):
        self.midi.save(filename=filename)




midis = os.listdir("MuseScoreMIDIS")
example = f"MuseScoreMIDIS/{midis[3]}"
print(example)
mid = MidiFile(example)
m = MidiTok(mid)
m.tokenize()
string = str(m)
print(string)
detok = MidiDeTok(string)
detok.parse()
detok.export('tryo.mid')
