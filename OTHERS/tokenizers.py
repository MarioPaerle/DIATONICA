from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
import os
from intomido.messages import Token

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

"""class MidiTok:
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
        self.str_value = []
        self.notes_on =  []
        self.notes_off = []
        self.notes = []
        self.implicit_transpose = KEY_IMPLICI_TRANSPOSE[self.key]
        self.events = dict()
        self.ticker = midi.ticks_per_beat
        self.all_ticker = self.ticker * self.numerator
        self.midi_track = midi.tracks
        self.n_tracks = len(self.midi_track)
        time = 0
        for msg in midi.merged_track:
            time += msg.time if len(self.notes) > 0 else 0
            if msg.type in ('note_on', 'note_off'):
                self.notes.append((time, msg))

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

    def _tokenize(self, transpose=False):
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
            string_on += f".{timeenc}.{abs_time}"

            for i, note in enumerate(notes_off):
                if i > 0:
                    string_off += "+"
                string_off += f"{note}"
            string_off += f".{timeenc}.{abs_time}" if len(string_off) > 0 else ""

            self.str_value_on.append(string_on)
            if len(string_off) > 0:
                self.str_value_off.append(string_off)

    def tokenize(self, transpose=False):
        if transpose:
            delta = self.implicit_transpose
        else:
            delta = 0

        for abs_time, note in self.notes:
            if abs_time in self.events:
                self.events[abs_time].append(note)
            else:
                self.events[abs_time] = [note]

        for abs_time in self.events:
            msgs = self.events[abs_time]
            timeenc = (abs_time % self.all_ticker) // self.denominator
            string = ""
            for i, msg in enumerate(msgs):
                if i > 0:
                    string += "+"
                string += f"{msg.note + delta}.{msg.velocity}"
            string += f"_{timeenc}_{abs_time}"

            self.str_value.append(string)

    def _str__(self):
        baseline = f"{self.clock}, {self.numerator}, {self.denominator}"
        return baseline + "\n" +  " ".join(self.str_value_on) + '\n' + " ".join(self.str_value_off)

    def __str__(self):
        baseline = f"{self.clock}, {self.numerator}, {self.denominator}"
        return baseline + "\n" +  " ".join(self.str_value)


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

        for msgs in self.notes_on.split():
            notes, time, abs_time = msgs.split('.')
            notes = notes.split('+')
            for i, note in enumerate(notes):
                if note == '':
                    continue
                if abs_time in self.events:
                    self.events[abs_time].append({"type":'note_on', "note" : int(note), "velocity" : 100})
                else:
                    self.events[abs_time]= [{"type":'note_on', "note" : int(note), "velocity" : 100}]

        for msgs in self.notes_off.split():
            notes, time, abs_time = msgs.split('.')
            notes = notes.split('+')
            for i, note in enumerate(notes):
                if note == '':
                    continue
                if abs_time in self.events:
                    self.events[abs_time].append({"type":'note_off', "note" : int(note), "velocity" : 100})
                else:
                    self.events[abs_time] = [{"type":'note_off', "note" : int(note), "velocity" : 100}]

        events = sorted(self.events.items(), key=lambda x: x[0])
        last_time = 0
        for event in events:
            for msg in event[1]:
                msg['time'] = (int(event[0]) - last_time if int(event[0]) > last_time else last_time) // numerator
                self.track.append(Message(**msg))
                last_time = int(event[0])

    def export(self, filename):
        self.midi.save(filename=filename)
"""

def binner(c):
    return round(c, -1)

class EasyTok:
    def __init__(self, midi: MidiFile):
        """The Tokenizer as implemented now, is specific for a midi file, parsed using Mido"""
        self.midi = midi
        for msg in midi:
            if msg.type == 'time_signature':
                self.numerator = msg.numerator
                self.denominator = msg.denominator
                self.musetime = f"{self.numerator}/{self.denominator}"
                self.clock = msg.clocks_per_click
            if msg.type == 'key_signature':
                self.key = msg.key


        self.ticker = midi.ticks_per_beat
        self.tokens = []

    def tokenize(self, transpose=False, explicit_delta=0, only_start=False):
        """Actually tokenizes the midi file.
        :param: transpose : it automatically transpose to C major if the scale is known
        :explicit_delta: it transpose the midi file up of explicit_delta semitones

        :return: the method returns a string which is the tokenized notes with numerator/denominator
        of the time signature before."""
        abs_time = 0
        string = f"{self.numerator}/{self.denominator}\n"
        delta = (KEY_IMPLICI_TRANSPOSE[self.key] if transpose else 0) + explicit_delta
        delta_time = 0
        for msg in self.midi.merged_track:
            time = msg.time
            abs_time += time
            if 'note' not in msg.type:
                delta_time += time
                continue

            if delta > 0:
                msg.note += delta

            if msg.velocity > 0 and not only_start:
                rel_time = (abs_time % (self.ticker * self.numerator)) // self.denominator
                rel_time = binner(rel_time)
                token = Token(msg, time + delta_time, rel_time)
                self.tokens.append(token)
                string += token.get_str(False) + " "

            delta_time = 0
        self.str_value = string
        return self.str_value

    def __str__(self):
        return self.str_value

    def __len__(self):
        return len(self.tokens)

    def shape(self):
        return len(self), len(self.tolist())

    def get_notes(self):
        return self.str_value.split('\n')[1]

    def tolist(self):
        musewords = [[]]
        loops = -1
        last_periodot = 720
        for token in self.get_notes().split(' '):
            if len(token) < 3:
                break
            periodot = token.split('-')[-1]
            if int(periodot) == 0 and last_periodot != periodot:
                loops += 1
                musewords.append([])

            musewords[loops].append(token)
            last_periodot = periodot

        return musewords[:-1]

class EasyDeTok:
    def __init__(self, string: str):
        """This Object reads the EasyTok type Encoding and detokenize it"""
        self.str = string
        self.events = dict()
        self.midi = MidiFile()
        self.track = MidiTrack()

    def decode(self):
        tempo, messages = self.str.split('\n')
        messages = messages.split()
        for msg in messages:
            note, velocity, time, relative_time = msg.split('-')
            note, velocity, time, relative_time = int(note), int(velocity), int(time), int(relative_time)
            self.track.append(Message(type='note_on', note=note, time=time, velocity=velocity))

    def export(self, filename='midioutput2.mid'):
        self.midi.tracks.append(self.track)
        self.midi.save(filename=filename)

if __name__ == '__main__':
    midis = os.listdir("MuseScoreMIDIS")
    print(len(midis))
    example = f"MuseScoreMIDIS/{midis[165]}"

    mid = MidiFile(example)
    tok = EasyTok(mid)
    tok.tokenize()
    string = str(tok)

    """
    print(string)
    print(len(string.split()))
    detok = EasyDeTok(string)
    detok.decode()
    detok.export()
    """

    print(tok.tolist())


