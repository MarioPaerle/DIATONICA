import pretty_midi
import numpy as np


def tokenize(
    midi_path: str,
    cycle_length_beats: int = 4,
    subdivisions_per_beat: int = 4
) -> list[str]:
    """
    Tokenize a piano-only MIDI file into a list of tokens. The first token is the time signature:
    "timesig.{numerator}.{denominator}", followed by note tokens "pitch.velocity.pos".

    Args:
        midi_path: Path to input MIDI file.
        cycle_length_beats: Beats per cycle (e.g., 4 for one bar in 4/4 time).
        subdivisions_per_beat: How many discrete ticks per beat (resolution).
    Returns:
        A list of tokens sorted by ascending start time, with an initial timesig token.
    """

    pm = pretty_midi.PrettyMIDI(midi_path)
    # extract first time signature change, default to 4/4
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        num, den = ts.numerator, ts.denominator
    else:
        num, den = 4, 4
    tokens: list[str] = [f"timesig.{num}.{den}"]

    # choose first non-drum instrument
    inst = next((i for i in pm.instruments if not i.is_drum), None)
    if inst is None:
        raise ValueError("No non-drum instrument found in MIDI.")

    beats = pm.get_beats()
    events: list[tuple[float, str]] = []

    past_cyc_pos = -1

    for note in inst.notes:
        idx = np.searchsorted(beats, note.start) - 1
        idx = max(0, min(idx, len(beats) - 2))
        dt = note.start - beats[idx]
        beat_dur = beats[idx + 1] - beats[idx]
        pos_beats = idx + dt / beat_dur

        total_ticks = int(round(pos_beats * subdivisions_per_beat))
        cycle_ticks = cycle_length_beats * subdivisions_per_beat
        cyc_pos = total_ticks % cycle_ticks

        if cyc_pos != past_cyc_pos:
            events.append((note.start, '<pos>'))

        token = f"{note.pitch}.{100 if note.velocity > 0 else 0}.{cyc_pos}"
        events.append((note.start, token))


        past_cyc_pos = cyc_pos
    for _, tok in sorted(events, key=lambda x: x[0]):
        tokens.append(tok)

    return tokens

def add_bars(tokens):
    last_pos = 100
    tokens2 = []
    for token in tokens:
        if '.' in token:
            _, _, pos = token.split(".")
        else:
            continue
        if int(pos) < last_pos:
            tokens2.append(f"<bar>")
        tokens2.append(token)
        last_pos = int(pos)

    return tokens2[1:]

def remove_bars(tokens):
    """This function will remove all the positional encoding tokens: like <bar>, <pos>"""

    lines2 = []
    for line in tokens:
        if '<bar>' in line or '<pos>' in line:
            continue
        lines2.append(line)
    return lines2


def save_tokens(tokens: list[str], filepath: str, putbars=False) -> None:
    """Save tokens to a text file, one per line."""
    if putbars:
        last_pos = 100
        tokens2 = []
        for token in tokens:
            _, _, pos = token.split(".")
            if int(pos) < last_pos:
                tokens2.append(f"<bar>")
            tokens2.append(token)
            last_pos = int(pos)

        tokens = tokens2[1:]

    with open(filepath, 'w') as f:
        f.write("\n".join(tokens))


def load_tokens(filepath: str, exclude_bars=False) -> list[str]:
    """Load tokens from a text file, one per line."""
    with open(filepath, 'r') as f:
        lines =  [line.strip() for line in f if line.strip()]
    if exclude_bars:
        lines2 = []
        for line in lines:
            if '<bar>' in line:
                continue
            lines2.append(line)
        return lines2
    else:
        return lines


def detokenize(
    tokens: list[str],
    output_path: str,
    cycle_length_beats: int = 4,
    subdivisions_per_beat: int = 4,
    default_duration_beats: float = 1.0,
    tempo: float = 120.0
) -> None:
    """
    Reconstruct a MIDI file from tokens. Expects first token "timesig.{num}.{den}".

    Args:
        tokens: List of tokens starting with time signature token.
        output_path: Path to write the reconstructed MIDI file.
        cycle_length_beats: Beats per cycle (must match tokenizer).
        subdivisions_per_beat: Ticks per beat (must match tokenizer).
        default_duration_beats: Duration of each note in beats.
        tempo: Constant tempo in BPM for output MIDI.
    """
    if not tokens:
        raise ValueError("Token list is empty")

    # parse time signature
    first = tokens[0]
    if not first.startswith('timesig.'):
        raise ValueError("First token must specify time signature, e.g. 'timesig.3.4'")
    _, sig = first.split('.', 1)
    num, den = map(int, sig.split('.'))

    pm = pretty_midi.PrettyMIDI()
    pm.time_signature_changes = [pretty_midi.TimeSignature(num, den, time=0)]
    pm._initial_tempo = tempo

    piano = pretty_midi.Instrument(program=0)
    seconds_per_beat = 60.0 / tempo
    cycle_count = 0
    prev_pos = -1
    cycle_ticks = cycle_length_beats * subdivisions_per_beat

    # process note tokens
    for tok in tokens[1:]:
        parts = tok.split('.')
        if len(parts) != 3:
            continue
        pitch, velocity, cyc_pos = map(int, parts)

        # detect wrap-around for cycles
        if prev_pos >= 0 and cyc_pos < prev_pos:
            cycle_count += 1
        prev_pos = cyc_pos

        # calculate start and end times
        start_beats = cycle_count * cycle_length_beats + (cyc_pos / subdivisions_per_beat)
        start_time = start_beats * seconds_per_beat
        end_time = start_time + default_duration_beats * seconds_per_beat

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=end_time
        )
        piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(output_path)

def transpose(tokens, k=0):
    retokens = []
    for token in tokens:
        if 'timesig' in token or 'bar' in token:
            retokens.append(token)
        else:
            pitch = int(token.split('.')[0]) + k
            retoken = f"{pitch}.{".".join(token.split('.')[1:])}"
            retokens.append(retoken)
    return retokens

def calm(tokens, k=0):
    retokens = []
    for token in tokens:
        if 'timesig' in token or 'bar' in token:
            retokens.append(token)
        else:
            pitch = int(token.split('.')[0]) + k
            retoken = f"{pitch}.{".".join(token.split('.')[1:])}"
            retokens.append(retoken)
    return retokens

import pypianoroll
import numpy as np
import os

def midi_to_segmented_pianorolls(midi_path: str, k: int, shift=0, div=1) -> list[np.ndarray]:
    if not isinstance(k, int) or k <= 0:
        print("Error: Segment length k must be a positive integer.")
        return []

    if not os.path.exists(midi_path):
        print(f"Error: MIDI file not found at {midi_path}")
        return []

    try:
        multitrack = pypianoroll.read(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file '{midi_path}': {e}")
        return []

    if not multitrack.tracks:
        print(f"No tracks found in MIDI file: {midi_path}")
        return []

    # Blend all tracks into a single pianoroll
    # 'sum' mode will sum the velocities of overlapping notes
    blended_pianoroll = multitrack.blend(mode='sum')

    original_pianoroll_shape = blended_pianoroll.shape

    if original_pianoroll_shape[0] == 0:
        print(f"The blended pianoroll in '{midi_path}' is empty (contains no notes).")
        return []

    # Create a dummy Track object to use its pad_to_multiple method
    # This is a workaround because blend() returns a numpy array, not a Track object.
    # We need to ensure the pianoroll has the necessary methods for padding.
    dummy_track = pypianoroll.Track(pianoroll=blended_pianoroll)
    dummy_track.pad_to_multiple(k)
    padded_pianoroll = dummy_track.pianoroll[0::div]

    num_timesteps = padded_pianoroll.shape[0]
    if num_timesteps == 0:
         print(f"Info: Pianoroll for '{midi_path}' (length {original_pianoroll_shape[0]}) became empty after padding attempt and division. This is unexpected if padding to k occurred correctly.")
         return []

    num_segments = num_timesteps // k
    segments = []

    if num_segments == 0:
        if num_timesteps > 0:
             print(f"Info: Pianoroll for '{midi_path}' (length {num_timesteps}) is shorter than segment length k={k} "
                   "even after padding, resulting in 0 full segments by floor division. "
                   "This is unexpected if padding to k occurred correctly.")
        return []

    for i in range(num_segments):
        segment = padded_pianoroll[i * k : (i + 1) * k, :]
        segments.append(np.roll(segment.T, shift, axis=0))

    return np.array(segments, dtype=np.uint8)

if __name__ == '__main__':
    segments = midi_to_segmented_pianorolls("MuseScoreMIDIS2/chet1004.mid", 128, 0, 2)
    print(segments.shape)
    import matplotlib.pyplot as plt
    plt.imshow(segments[0])
    plt.show()

