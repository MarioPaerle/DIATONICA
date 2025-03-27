from midiutil import MIDIFile
from intomido.messages import Message
from mido import MidiFile, MidiTrack
import mido

class BasicComposer:
    def __init__(self, tempo=120, channel=0, volume=100, program=0):
        """
        Initialize the Composer Object:
        :param tempo: int, BPM of the midi
        :param channel: int, channel of the midi
        :param volume: int, volume of the midi
        """
        self.midi = MIDIFile(1)  # Un file MIDI con una sola traccia
        self.midi.addTempo(0, 0, tempo)
        self.channel = channel
        self.volume = volume
        self.current_time = 0.0  # Tempo corrente (in beats)
        self.midi.addProgramChange(0, channel, 0, program)

    def _parse_timestep(self, timestep):
        """
        Parses the timestep string for example "1/4" or "6/8"
        """
        try:
            num, denom = map(int, timestep.split('/'))
        except Exception as e:
            raise ValueError("Wrong format'") from e

        return 4 * (num / denom)

    def add_notes_fixed_timestep(self, notes, timestep='1/4', mode='tie', repeataslong=False):
        """
        Adds a sequence of notes to the MIDIFile object

        Parmeters:
          :param notes: List of notes
                   * a single numerical value (must convert to int) which is the midi scale pitch of the note;
                   * a single character, supported are: "-" which means repeat past note, and "_" which means pause.
          :param timestep: a single note duration, which is fixed!
        """
        base_duration = self._parse_timestep(timestep)
        i = 0

        if repeataslong:
            notes2 = []
            for i, note in enumerate(notes):
                if i > 0 and note == notes[i-1]:
                    notes2.append('-')
                else:
                    notes2.append(note)

            notes = notes2
        ct = self.current_time
        while i < len(notes):
            elem = notes[i]
            if elem == '-':
                ct += base_duration
                i += 1
            else:
                try:
                    pitch = int(elem)
                except Exception as e:
                    raise ValueError(f"Valore nota non valido: {elem}") from e

                total_duration = base_duration
                if mode == 'tie':
                    # Controlla se le note successive sono trattini e, in tal caso, estendi la durata
                    while (i + 1 < len(notes)) and (notes[i + 1] == '-'):
                        total_duration += base_duration
                        i += 1
                # Aggiungi la nota con la durata totale calcolata
                self.midi.addNote(track=0, channel=self.channel, pitch=pitch,
                                  time=ct, duration=total_duration,
                                  volume=self.volume)
                ct += total_duration
                i += 1

    def tomidi(self, filename):
        """
        Salva il file MIDI nel percorso specificato.
        """
        with open(filename, "wb") as f:
            self.midi.writeFile(f)


class Composer:
    def __init__(self, ticks_per_beat=480, tempo=500000):
        # Ticks per beat e tempo (microsecond per beat) usati per calcolare le durate
        self.ticks_per_beat = ticks_per_beat
        self.tempo = tempo
        self.messages = []  # lista di Message personalizzati
        self.current_time_a = 0  # tempo corrente per la modalità 'append'

    def _get_step_ticks(self, step):
        """Converte una stringa di step (es. '1/8') in ticks"""
        num, denom = map(int, step.split('/'))
        # Calcola i tick per step: partiamo dal presupposto che 1/4 equivale a 1 beat (ticks_per_beat)
        beat_fraction = num / denom
        # Se 1/4 = ticks_per_beat allora 1/8 = ticks_per_beat/2, ecc.
        return int(self.ticks_per_beat * (beat_fraction / 0.25))

    def add_fv_pattern(self, pattern, step='1/8', mode='a', channel=0):
        """Adds a note pattern:
          - A number indicates a note.
          - '-' indicates to repeat the previous note.
          - '_' is interpreted as a pause.
        The 'mode' parameter:
          - 'a': appends after the last message (cumulative timeline modification)
          - 'o': overlays (uses time=0 as base)"""

        step_ticks = self._get_step_ticks(step)
        # Determina il punto di partenza in base alla modalità
        if mode == 'a':
            start_time = self.current_time_a
        else:  # mode 'o' (overlay)
            start_time = 0

        last_note = None
        time_cursor = start_time
        for elem in pattern:
            if elem == '_':
                time_cursor += step_ticks
                continue
            if elem == '-' and last_note is not None:
                note = last_note
            elif isinstance(elem, int):
                note = elem
                last_note = note
            else:
                # no - so skip
                continue

            # Crea il messaggio NOTE_ON in corrispondenza del time_cursor
            msg_on = Message(note=note, time=time_cursor, velocity=64, action='on', channel=channel)
            self.messages.append(msg_on)
            msg_off = Message(note=note, time=time_cursor + step_ticks, velocity=64, action='off', channel=channel)
            self.messages.append(msg_off)
            time_cursor += step_ticks

        # Se la modalità è 'a', aggiorna il current_time_a per future aggiunte
        if mode == 'a':
            self.current_time_a = time_cursor

    def finalize(self, filename):
        """Sorts messages by absolute time, converts them to delta time, and creates the MIDI file."""

        # Ordina per tempo assoluto; in caso di parità, mettiamo prima i note_on
        self.messages.sort(key=lambda m: (m.time, 0 if m.action == 'on' else 1))

        track = MidiTrack()
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        mid.tracks.append(track)
        # Imposta il tempo
        track.append(mido.MetaMessage('set_tempo', tempo=self.tempo, time=0))

        # Trasforma il tempo assoluto in delta time
        last_time = 0
        for m in self.messages:
            delta = m.time - last_time
            last_time = m.time
            midi_msg = m.tomido(time=delta)
            track.append(midi_msg)

        # Salva il file MIDI
        mid.save(filename)