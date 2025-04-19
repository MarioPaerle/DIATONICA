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

    for note in inst.notes:
        idx = np.searchsorted(beats, note.start) - 1
        idx = max(0, min(idx, len(beats) - 2))
        dt = note.start - beats[idx]
        beat_dur = beats[idx + 1] - beats[idx]
        pos_beats = idx + dt / beat_dur

        total_ticks = int(round(pos_beats * subdivisions_per_beat))
        cycle_ticks = cycle_length_beats * subdivisions_per_beat
        cyc_pos = total_ticks % cycle_ticks

        token = f"{note.pitch}.{100 if note.velocity > 0 else 0}.{cyc_pos}"
        events.append((note.start, token))

    for _, tok in sorted(events, key=lambda x: x[0]):
        tokens.append(tok)

    return tokens

def add_bars(tokens):
    last_pos = 100
    tokens2 = []
    for token in tokens:
        _, _, pos = token.split(".")
        if int(pos) < last_pos:
            tokens2.append(f"<bar>")
        tokens2.append(token)
        last_pos = int(pos)

    return tokens2[1:]

def remove_bars(tokens):
    lines2 = []
    for line in tokens:
        if '<bar>' in line:
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

    # setup PrettyMIDI with time signature and tempo
    pm = pretty_midi.PrettyMIDI()
    # add time signature event at time 0
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



if __name__ == '__main__':
    tokens = tokenize('MuseScoreMIDIS/chpn_op25_e12_format0.mid', cycle_length_beats=4, subdivisions_per_beat=4)
    # save_tokens(tokens, 'tokens.txt', putbars=True)
    """tokens = load_tokens('tokens.txt', exclude_bars=True)
    detokenize(tokens, 'output.mid', tempo=120.0, cycle_length_beats=4, subdivisions_per_beat=4)"""
