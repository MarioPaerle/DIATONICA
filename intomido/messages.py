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


class Token:
    def __init__(self, msg: Message, time, pos_em):
        self.time = time
        self.underline = msg
        self.pos_em = pos_em

    def get_mido(self):
        return self.underline

    def get_str(self, time=True):
        note = self.underline.note
        velocity = self.underline.velocity
        if velocity > 0:
            velocity = 100
        if time:
            return f"{note}-{velocity}-{self.time}-{self.pos_em}"
        else:
            return f"{note}-{velocity}-{self.pos_em}"

