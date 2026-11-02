"""Simple piano-like tone synthesis, used for tests and demos."""

from __future__ import annotations

import numpy as np

from .notes import midi_to_freq, name_to_midi


def piano_tone(midi: int, duration: float, sr: int = 44100, a4: float = 440.0,
               n_harmonics: int = 20, inharmonicity: float = 3e-4,
               amplitude: float = 0.2, seed: int | None = None) -> np.ndarray:
    """A piano-ish tone: stretched (inharmonic) partials, decaying envelopes,
    with higher partials quieter and dying faster."""
    rng = np.random.default_rng(seed if seed is not None else midi)
    f0 = midi_to_freq(midi, a4)
    t = np.arange(int(duration * sr)) / sr
    out = np.zeros_like(t)
    # low strings: weaker fundamental relative to overtones
    fund_gain = 0.35 if f0 < 80 else 1.0
    for h in range(1, n_harmonics + 1):
        fh = h * f0 * np.sqrt(1 + inharmonicity * h * h)
        if fh >= sr / 2 - 100:
            break
        amp = (fund_gain if h == 1 else 1.0) / h ** 1.1
        amp *= rng.uniform(0.6, 1.0)
        decay = 1.5 + 0.6 * h + f0 / 400
        out += amp * np.exp(-decay * t) * np.sin(2 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))
    attack = np.minimum(1.0, t / 0.005)
    out *= attack
    return amplitude * out / (np.max(np.abs(out)) + 1e-12)


def chord(notes, duration: float = 2.0, sr: int = 44100, **kw) -> np.ndarray:
    """Mix piano tones for note names like ``["C4", "E4", "G4"]``."""
    x = sum(piano_tone(name_to_midi(n), duration, sr, **kw) for n in notes)
    return x / max(1.0, np.max(np.abs(x)) / 0.9)


def sequence(chords, duration: float = 1.0, sr: int = 44100, **kw) -> np.ndarray:
    """Concatenate several chords, each lasting ``duration`` seconds."""
    return np.concatenate([chord(c, duration, sr, **kw) for c in chords])
