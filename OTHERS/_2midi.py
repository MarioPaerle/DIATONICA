from mido import Message, MidiFile, MidiTrack
import numpy as np
import os


class MidiTok:
    def __init__(self, midi: MidiFile):
        self.midi = midi
        for msg in midi:
            if msg.type == 'time_signature':
                self.numerator = msg.numerator
                self.denominator = msg.denominator
                self.musetime = f"{self.numerator}/{self.denominator}"
                self.clock = msg.clocks_per_click

        self.value = []
        self.notes_on = dict()
        self.notes_off = dict()
        time = 0
        for msg in midi.merged_track:
            time += msg.time
            if msg.type == 'note_on':
                self.notes_on[time] = msg
            elif msg.type == 'note_off':
                self.notes_off[time] = msg

        self.orderes_notes_on =  sorted(self.notes_on.items(), key=lambda x: x[0])
        self.orderes_notes_off = sorted(self.notes_off.items(), key=lambda x: x[0])

    def tokenize(self):
        for abs_time, note in self.orderes_notes_on:
            print(abs_time, note)
            pass

midis = os.listdir("MuseScoreMIDIS")
mid = MidiFile(f"MuseScoreMIDIS/{midis[40]}")
m = MidiTok(mid)
m.tokenize()


