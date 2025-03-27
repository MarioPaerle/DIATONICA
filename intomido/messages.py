import mido
from mido import MidiFile, MidiTrack


class Message:
    def __init__(self, note, time=100, velocity=100, action='on', channel=0):
        """Generate the Midi Message Object"""
        self.note = note
        self.velocity = velocity
        self.time = time  # tempo assoluto (tick)
        self.action = action  # 'on' o 'off'
        self.channel = channel

    def __str__(self):
        return f"MESSAGE<{self.action}, note: {self.note}, velocity: {self.velocity}, time: {self.time}, channel: {self.channel}>"

    def __repr__(self):
        return self.__str__()

    def tomido(self, time=None):
        return mido.Message('note_'+self.action, note=self.note, velocity=self.velocity, time=self.time if time is None else time, channel=self.channel)

