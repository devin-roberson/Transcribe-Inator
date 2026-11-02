"""Transcribe-Inator: detect the notes being played in a .wav file."""

from .analysis import DetectedNote, Segment, Settings, detect_notes, detect_onsets, transcribe
from .audio import load_wav
from .notes import midi_to_freq, midi_to_name, name_to_midi

__all__ = [
    "DetectedNote", "Segment", "Settings", "detect_notes", "detect_onsets",
    "transcribe", "load_wav", "midi_to_freq", "midi_to_name", "name_to_midi",
]
__version__ = "0.1.0"
