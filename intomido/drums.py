from composers import *
import numpy as np

class PercussionPattern:
    def __init__(self, duration, subdivision):
        self.subdivision = subdivision
        self.duration = duration
        self.pattern = np.zeros((duration*subdivision)).tolist()
        self.velocity_levels = [120, 100, 80, 60, 40, 20, 10, 0]

    def fill(self, *mods):
        if len(mods) > len(self.velocity_levels):
            raise ValueError("Not enough velocity levels for the specified mods")
        for i in range(len(self.pattern)):
            for mod in mods:
                pass
    def __mul__(self, other):
        self.pattern.extend(other.pattern)
        return self


class DrumMachine:
    pass

