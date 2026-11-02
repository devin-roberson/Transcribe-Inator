"""Helpers for converting between frequencies, MIDI numbers and note names."""

from __future__ import annotations

import math
import re

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                  "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

PIANO_LOW = 21   # A0
PIANO_HIGH = 108  # C8


def midi_to_freq(midi: float, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((midi - 69) / 12.0)


def freq_to_midi(freq: float, a4: float = 440.0) -> float:
    return 69 + 12 * math.log2(freq / a4)


def midi_to_name(midi: int) -> str:
    """60 -> 'C4'."""
    midi = int(round(midi))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    """'C4' -> 60, 'Bb3' -> 58, 'F#5' -> 78."""
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)(-?\d+)\s*", name)
    if not m:
        raise ValueError(f"Not a note name: {name!r}")
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pitch = letter + accidental
    octave_shift = 0
    if pitch in _FLAT_TO_SHARP:
        new = _FLAT_TO_SHARP[pitch]
        if pitch == "Cb":
            octave_shift = -1
        elif pitch == "B#":
            octave_shift = 1
        pitch = new
    return NOTE_NAMES.index(pitch) + 12 * (octave + 1 + octave_shift)
