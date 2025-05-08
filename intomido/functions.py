import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pypianoroll
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io.wavfile import write


def midi_to_numpy(midi_path: str, fs: int = 100) -> np.ndarray:
    """
    Convert a MIDI file into a piano roll numpy array.

    Parameters:
      midi_path (str): Path to the MIDI file (e.g. from the Maestro dataset).
      fs (int): Sampling frequency in frames per second. Default is 100.

    Returns:
      np.ndarray: A piano roll array of shape (128, T), where T = int(midi_duration * fs).
                  Each column represents a time step of 1/fs seconds; the values are note velocities.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    # piano_roll = piano_roll.astype(np.uint8)
    return piano_roll


def mod_to_midi_representation(jsb_array: np.ndarray, low_pitch: int) -> np.ndarray:
    result = []
    if len(jsb_array.shape) == 3:
        for i in range(jsb_array.shape[0]):
            result.append(mod_to_midi_representation(jsb_array[i], low_pitch))
    else:
        result  = np.zeros((128, jsb_array.shape[0]))
        for i in range(jsb_array.shape[0]):
            for j in range(jsb_array.shape[1]):
                if jsb_array[i,j] != 0:
                    result[jsb_array[i,j] - low_pitch, i] = 1
    return result

def plotmidi(midi):
    track = pypianoroll.Track(pianoroll=midi.T)
    multitrack = pypianoroll.Multitrack(tracks=[track])
    fig, ax = plt.subplots(figsize=(10, 4))
    pypianoroll.plot_multitrack(multitrack, axs=[ax])
    plt.show()

def midi_to_audio(midi_path, fs=44100, outputfile="output.wav", tempo=100):
    if isinstance(midi_path, str):
        midi_data = pretty_midi.PrettyMIDI(midi_path, initial_tempo=tempo)
    else:
        midi_data = midi_path
    audio = midi_data.synthesize(fs=fs)
    scaled_audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(outputfile, 44100, scaled_audio)
    return audio

def midi_to_audio_fluidsynth(midi_path, sf2_path, fs=44100):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    audio = midi_data.fluidsynth(fs=fs, sf2_path=sf2_path)
    return audio

def roll(X, W, axis=None):
    return sliding_window_view(X, window_shape=W, axis=axis)


def multi_hot_to_midi(piano_roll: np.ndarray, time_per_step: float = 0.2,
                      velocity: int = 100) -> pretty_midi.PrettyMIDI:
    """
    Convert a multi-hot encoded piano roll (2D NumPy array with shape (T, 128)) into a PrettyMIDI object.

    Parameters:
      piano_roll (np.ndarray): 2D array of shape (T, 128) where each row is a binary (or multi-hot) vector.
      time_per_step (float): Duration (in seconds) of each time step. Default is 0.05 sec.
      velocity (int): Velocity for note on events. Default is 100.

    Returns:
      pretty_midi.PrettyMIDI: A MIDI object representing the piano roll.
    """
    T, n_pitches = piano_roll.shape
    if n_pitches != 128:
        raise ValueError("The input piano roll must have 128 columns (for MIDI notes 0-127).")

    # Create a new PrettyMIDI object and a single instrument (Acoustic Grand Piano)
    midi_obj = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Dictionary to keep track of active notes: pitch -> start time
    active_notes = {}

    # Iterate over time steps
    for t in range(T):
        current_time = t * time_per_step
        current_frame = piano_roll[t]  # shape: (128,)

        for pitch in range(128):
            is_active = current_frame[pitch] > 0

            # Check previous state: if first time step, assume note was off.
            prev_active = piano_roll[t - 1][pitch] > 0 if t > 0 else False

            # Note-on: the note is now active but wasn't active in the previous step.
            if is_active and not prev_active:
                active_notes[pitch] = current_time
            # Note-off: the note was active in the previous step but is now off.
            elif not is_active and prev_active:
                start_time = active_notes.pop(pitch, current_time)
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=current_time)
                instrument.notes.append(note)

    # Close any notes still active at the end of the piano roll
    final_time = T * time_per_step
    for pitch, start_time in active_notes.items():
        note = pretty_midi.Note(velocity=0, pitch=pitch, start=start_time, end=final_time)
        instrument.notes.append(note)

    midi_obj.instruments.append(instrument)
    return midi_obj

def add_random_holes(X, p=0.1):
    mask = np.random.rand(*X.shape) > p  # True with probability (1-p)
    return X * mask.astype(X.dtype)


def torch_imshow(X, index=0, channels=1):
    if channels == 1:
        plt.imshow(X.cpu().numpy()[index, 0, :, :])
        plt.show()

def np_imshow(X, index=0, channels=1, ncwh=True):
    if ncwh:
        if channels == 1:
            plt.imshow(X[index, 0, :, :])
            plt.show()
    else:
        if channels == 1:
            plt.imshow(X[index, :, :])
            plt.show()


pm = pretty_midi
def pm_swing(
    pm,
    cycle_length_beats: int = 9,
    subdivisions_per_beat: int = 6):
    inst = next((i for i in pm.instruments if not i.is_drum), None)
    if inst is None:
        raise ValueError("No non-drum instrument found in MIDI.")

    beats = pm.get_beats()
    events: list[tuple[float, str]] = []

    for note in inst.notes:
        idx = np.searchsorted(beats, note.start) - 1
        idx = max(0, min(idx, len(beats) - 2))
        dt = note.start - beats[idx]
        beat_dur = beats[idx + 1] - beats[idx]
        pos_beats = idx + dt / beat_dur

        total_ticks = int(round(pos_beats * subdivisions_per_beat))
        cycle_ticks = cycle_length_beats * subdivisions_per_beat
        cyc_pos = total_ticks % cycle_ticks
        print(cyc_pos)
        if cyc_pos % 2 == 1 and note.velocity > 0:
            note.start += 0.02
            note.end   += 0.02


def nearest(value, candidates):
    if not candidates:
        raise ValueError("`candidates` must contain at least one value.")
    return min(candidates, key=lambda x: abs(x - value))
