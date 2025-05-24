import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import pypianoroll
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io.wavfile import write
from IPython.display import Audio



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

def midi_to_audio(midi_path, fs=44100, tempo=100, autoplay=False):
    from IPython.display import Audio
    pm = midi_path

    wav = pm.synthesize(fs)
    wav = wav / np.max(np.abs(wav))
    return Audio(wav, rate=fs, autoplay=True)

import pretty_midi
import numpy as np
import sounddevice as sd
import os

def play_midi_audio(midi_input, fs=44100, play=True):
    try:
        if isinstance(midi_input, str) and os.path.exists(midi_input):
            midi_data = pretty_midi.PrettyMIDI(midi_input)
        elif isinstance(midi_input, pretty_midi.PrettyMIDI):
            midi_data = midi_input
        else:
            raise TypeError("Input deve essere un percorso a un file .mid o un oggetto PrettyMIDI.")

        # Sintetizza l'audio
        audio = midi_data.synthesize(fs=fs)

        if np.isnan(audio).any() or np.max(np.abs(audio)) == 0:
            raise ValueError("Audio contiene NaN o è silenzioso.")

        audio /= np.max(np.abs(audio))
        if play:
            sd.play(audio, samplerate=fs)
            sd.wait()
        return audio


    except Exception as e:
        print(f"Errore durante la riproduzione: {e}")



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
                velocity = piano_roll[t - 1][pitch].item()
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


def nearest_numpy(value: np.ndarray, candidates):
    if not candidates:
        raise ValueError("`candidates` must contain at least one value.")

    candidates = np.asarray(candidates)

    abs_diffs = np.abs(value[:, :, None] - candidates[None, None, :])

    nearest_candidate_indices = np.argmin(abs_diffs, axis=2)

    nearest_values = candidates[nearest_candidate_indices]

    return nearest_values


def cast_pianoroll_to_scale(pianoroll: np.ndarray, scale_notes):
    """
    Casts each active note in a pianoroll to the nearest note in a given scale.

    Args:
        pianoroll: A 2D NumPy array representing a pianoroll (time_steps, 128).
                   Values are velocities.
        scale_notes: A list or NumPy array of MIDI note numbers representing the
                     scale. Must not be empty.

    Returns:
        A new NumPy array representing the modified pianoroll, where velocities
        have been moved to the nearest scale notes. The shape and dtype are
        the same as the input pianoroll.

    Raises:
        ValueError: If `scale_notes` is empty or if the pianoroll shape is
                    unexpected (e.g., not 2D or second dim not 128).
    """
    if not scale_notes:
        raise ValueError("`scale_notes` must contain at least one value.")

    scale_notes = np.asarray(scale_notes, dtype=int)

    if pianoroll.ndim != 2:
        raise ValueError("Pianoroll must be a 2D array.")

    num_time_steps, num_notes = pianoroll.shape
    if num_notes != 128:
        print(f"Warning: Pianoroll second dimension is {num_notes}, expected 128.")

    output_pianoroll = np.zeros_like(pianoroll)

    if np.issubdtype(pianoroll.dtype, np.integer):
        max_velocity = 127
    else:  # Assuming float
        max_velocity = 1.0

    all_possible_notes = np.arange(128)
    abs_diffs = np.abs(scale_notes[:, np.newaxis] - all_possible_notes)
    nearest_scale_note_indices = np.argmin(abs_diffs, axis=0)
    mapping_to_nearest_scale_note = scale_notes[nearest_scale_note_indices]

    # Now process each time step
    for t in range(num_time_steps):
        pianoroll_slice = pianoroll[t, :]

        active_note_indices = np.where(pianoroll_slice > 0)[0]

        if active_note_indices.size == 0:
            continue

        active_velocities = pianoroll_slice[active_note_indices]

        # Find the target scale note for each active original note using the pre-calculated mapping
        target_scale_notes_for_active = mapping_to_nearest_scale_note[active_note_indices]
        output_slice = output_pianoroll[t, :]  # Get the slice to modify in place
        np.add.at(output_slice, target_scale_notes_for_active, active_velocities)

        np.clip(output_slice, 0, max_velocity, out=output_slice)

    return output_pianoroll

def minimize_rivolt(a:list, b:list):
    """This returns a list of notes b which minimizes the distance from notes in a"""
    errors_configurations = {}

    for i in range(3):
        errors_configurations[i] = mse_list(a, rivolt(b, i))

    conf_min = min(errors_configurations.keys(), key=lambda x: errors_configurations[x])
    return rivolt(b, conf_min)


def rivolt(a, k):
    c = a.copy()
    for i in range(k):
        c[i] = c[i] + 12

    return sorted(c)

def mse_list(a, b):
    a = [aa.note for aa in a]
    b = [bb.note for bb in b]
    return sum([(aa - bb)**2 for aa, bb in zip(a, b)])
