"""Polyphonic note detection from audio via spectral analysis.

The approach, in short:

1. **Spectrum** – a Hann-windowed, zero-padded FFT of the audio segment.
2. **Peak picking** – local maxima of the magnitude spectrum, refined with
   parabolic interpolation to get sub-bin frequency/amplitude estimates.
3. **Harmonic salience** – every candidate note (A0..C8) is scored by summing
   the (weighted) amplitudes of spectral peaks found at its harmonic
   positions ``f0, 2*f0, 3*f0, ...``. Partial positions are tracked
   progressively so the slightly stretched overtones of real piano strings
   (inharmonicity) are still matched.
4. **Iterative estimation & cancellation** – the best-scoring note is accepted,
   its harmonics are subtracted from the peak list (using a smoothed spectral
   envelope, so harmonics shared with other notes, e.g. an octave above, are
   only partly removed), and the process repeats until the remaining salience
   is too weak relative to the first note found.

A candidate must have its fundamental (or both its 2nd and 3rd harmonics)
present, which guards against the classic "octave below" / "fifth below"
errors where a non-played note matches a subset of another note's overtones.

For recordings with several chords/notes in sequence, :func:`detect_onsets`
finds note attacks via spectral flux and :func:`transcribe` analyses each
segment between attacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .notes import PIANO_HIGH, PIANO_LOW, midi_to_freq, midi_to_name


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class DetectedNote:
    midi: int
    name: str
    frequency: float      # measured fundamental (Hz), or nominal if missing
    strength_db: float    # salience relative to the strongest note (0 dB)
    cents_off: float      # deviation of measured f0 from the tuned pitch

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "midi": self.midi,
            "frequency": round(self.frequency, 2),
            "strength_db": round(self.strength_db, 1),
            "cents_off": round(self.cents_off, 1),
        }


@dataclass
class Segment:
    start: float
    end: float
    notes: list[DetectedNote] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class Settings:
    a4: float | None = None          # reference tuning; None = auto-estimate
    max_notes: int = 8               # maximum polyphony per segment
    threshold: float = 0.2           # min salience relative to first note
    peak_floor_db: float = -60.0     # ignore spectral peaks below max - this
    max_harmonics: int = 20
    cancel_gain: float = 0.8         # how much of a note's envelope to cancel
    fmax: float = 6000.0             # ignore partials above this frequency
    low_note: int = PIANO_LOW
    high_note: int = PIANO_HIGH


# --------------------------------------------------------------------------- #
# Spectrum & peaks
# --------------------------------------------------------------------------- #
def magnitude_spectrum(x: np.ndarray, sr: int, min_fft: int = 1 << 16):
    """Return ``(freqs, magnitude)`` of a Hann-windowed, zero-padded FFT."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("Empty audio segment")
    window = np.hanning(n)
    nfft = max(min_fft, 2 << int(math.ceil(math.log2(n))))
    spec = np.abs(np.fft.rfft(x * window, nfft))
    spec /= window.sum() / 2.0  # a full-scale sine -> magnitude ~1
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, spec


def spectral_peaks(freqs: np.ndarray, mag: np.ndarray, floor_db: float = -60.0,
                   fmin: float = 20.0, fmax: float = 8000.0,
                   noise_win_hz: float = 200.0, noise_margin_db: float = 15.0):
    """Find spectral peaks; return arrays ``(peak_freqs, peak_amps)``.

    Peak positions/amplitudes are refined with parabolic interpolation on the
    log-magnitude spectrum.
    """
    eps = 1e-12
    db = 20 * np.log10(mag + eps)
    top = db.max()
    lo = int(np.searchsorted(freqs, fmin))
    hi = int(np.searchsorted(freqs, fmax))
    idx, _ = find_peaks(db[lo:hi], height=top + floor_db, prominence=6.0)
    idx = idx + lo
    # Adaptive noise floor: keep peaks well above the local median level
    # (estimated on a coarse grid to keep it cheap).
    step = max(1, int(round(noise_win_hz / 8 / (freqs[1] - freqs[0]))))
    coarse = db[::step]
    k = max(1, int(round(noise_win_hz / (freqs[step] - freqs[0]))))
    floor = median_filter(coarse, size=k | 1, mode="nearest")
    idx = idx[db[idx] > floor[np.minimum(idx // step, len(floor) - 1)] + noise_margin_db]
    idx = idx[(idx > 0) & (idx < len(mag) - 1)]
    if len(idx) == 0:
        return np.empty(0), np.empty(0)
    a, b, c = db[idx - 1], db[idx], db[idx + 1]
    denom = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(denom != 0, 0.5 * (a - c) / denom, 0.0)
    p = np.clip(p, -0.5, 0.5)
    df = freqs[1] - freqs[0]
    pf = freqs[idx] + p * df
    pdb = b - 0.25 * (a - c) * p
    return pf, 10 ** (pdb / 20)


def estimate_tuning(peak_freqs: np.ndarray, peak_amps: np.ndarray,
                    n_peaks: int = 12, fmax: float = 1500.0) -> float:
    """Estimate the reference A4 from the strongest low-frequency peaks.

    Uses an amplitude-weighted circular mean of each peak's deviation (in
    cents) from the nearest equal-tempered pitch at A4 = 440 Hz. Returns the
    estimated A4 in Hz (clamped to within +-50 cents of 440).
    """
    mask = (peak_freqs > 0) & (peak_freqs < fmax)
    f, a = peak_freqs[mask], peak_amps[mask]
    if len(f) == 0:
        return 440.0
    order = np.argsort(a)[::-1][:n_peaks]
    f, a = f[order], a[order]
    semis = 12 * np.log2(f / 440.0)
    angle = 2 * np.pi * (semis - np.round(semis))
    z = np.sum(a * np.exp(1j * angle))
    if abs(z) < 1e-12:
        return 440.0
    dev_cents = 100 * np.angle(z) / (2 * np.pi)
    return 440.0 * 2 ** (dev_cents / 1200)


# --------------------------------------------------------------------------- #
# Multi-pitch estimation
# --------------------------------------------------------------------------- #
def _harmonic_weight(f0: float, h: int) -> float:
    # Klapuri (2006) weighting: favours lower partials / lower-pitched notes
    # just enough to avoid octave-above errors.
    return (f0 + 27.0) / (h * f0 + 320.0)


def _cents(f: np.ndarray | float, ref: float):
    return 1200 * np.log2(np.asarray(f) / ref)


def _match_harmonics(f0: float, pf: np.ndarray, pa: np.ndarray,
                     settings: Settings):
    """Find the peaks that belong to a candidate fundamental ``f0``.

    Returns a list of ``(h, peak_index)`` tuples.
    """
    matches = []
    if len(pf) == 0:
        return matches
    ref_f, ref_h = f0, 1   # last confirmed partial; used to follow stretching
    misses = 0
    for h in range(1, settings.max_harmonics + 1):
        pred = ref_f * h / ref_h
        if pred > settings.fmax:
            break
        tol = 45.0 if h == 1 else 30.0
        dev = np.abs(_cents(pf, pred))
        j = int(np.argmin(dev))
        if dev[j] <= tol:
            matches.append((h, j))
            ref_f, ref_h = pf[j], h
            misses = 0
        else:
            misses += 1
            if misses >= 4:
                break
    return matches


def _salience(f0: float, matches, pa: np.ndarray) -> float:
    return sum(_harmonic_weight(f0, h) * pa[j] for h, j in matches)


def _plausible(f0: float, matches) -> bool:
    """Reject candidates that only match a subset of another note's partials.

    The fundamental must be present (even weak bass fundamentals normally
    show up as a peak, just a quiet one).
    """
    return any(h == 1 for h, _ in matches)


def _partial_envelope(matches, pa: np.ndarray) -> dict[int, float]:
    """Robustly fit a smooth ``c * h**-p`` envelope to a note's partials.

    Partials boosted because another note shares them lie above the fit, so
    subtracting only the envelope leaves their excess for the other note.
    """
    hs = np.array([h for h, _ in matches], dtype=float)
    la = np.log(np.array([pa[j] for _, j in matches]) + 1e-12)
    lh = np.log(hs)
    best = None
    for p in np.linspace(0.0, 3.0, 31):
        c = np.median(la + p * lh)
        err = np.median(np.abs(la - (c - p * lh)))
        if best is None or err < best[0]:
            best = (err, c, p)
    _, c, p = best
    return {int(h): float(np.exp(c - p * np.log(h))) for h in hs}


def detect_notes_in_peaks(pf: np.ndarray, pa: np.ndarray,
                          settings: Settings | None = None,
                          a4: float = 440.0) -> list[DetectedNote]:
    """Iteratively pick notes from a set of spectral peaks."""
    settings = settings or Settings()
    orig = pa
    pa = pa.copy()
    found: list[DetectedNote] = []
    used: set[int] = set()
    first_sal = None
    candidates = range(settings.low_note, settings.high_note + 1)

    for _ in range(settings.max_notes):
        best = None
        for m in candidates:
            if m in used:
                continue
            f0 = midi_to_freq(m, a4)
            matches = _match_harmonics(f0, pf, orig, settings)
            if not _plausible(f0, matches):
                continue
            s = _salience(f0, matches, pa)
            if best is None or s > best[0]:
                best = (s, m, matches)
        if best is None:
            break
        sal, m, matches = best
        if first_sal is None:
            first_sal = sal
        elif sal < settings.threshold * first_sal:
            break

        f0_nom = midi_to_freq(m, a4)
        h1 = [j for h, j in matches if h == 1]
        measured = float(pf[h1[0]]) if h1 else f0_nom
        found.append(DetectedNote(
            midi=m,
            name=midi_to_name(m),
            frequency=measured,
            strength_db=float(20 * np.log10(sal / first_sal)),
            cents_off=float(_cents(measured, f0_nom)),
        ))
        used.add(m)

        # Cancel this note's partials, removing at most a smooth envelope's
        # worth from each so that partials shared with other notes survive.
        env = _partial_envelope(matches, pa)
        for h, j in matches:
            if h == 1:
                pa[j] = 0.0
            else:
                pa[j] = max(0.0, pa[j] - settings.cancel_gain * env[h])

    found.sort(key=lambda n: n.midi)
    return found


def detect_notes(x: np.ndarray, sr: int, settings: Settings | None = None):
    """Detect the notes sounding in an audio excerpt.

    Returns ``(notes, a4)`` where ``a4`` is the tuning reference used.
    """
    settings = settings or Settings()
    freqs, mag = magnitude_spectrum(x, sr)
    pf, pa = spectral_peaks(freqs, mag, floor_db=settings.peak_floor_db,
                            fmin=20.0, fmax=min(settings.fmax, sr / 2 - 1))
    a4 = settings.a4 if settings.a4 else estimate_tuning(pf, pa)
    return detect_notes_in_peaks(pf, pa, settings, a4), a4


# --------------------------------------------------------------------------- #
# Onsets & full transcription
# --------------------------------------------------------------------------- #
def detect_onsets(x: np.ndarray, sr: int, frame: int = 2048, hop: int = 512,
                  delta: float = 0.08, min_gap: float = 0.12) -> list[float]:
    """Return note-attack times (seconds) using log spectral flux."""
    if len(x) < frame:
        return [0.0]
    n_frames = 1 + (len(x) - frame) // hop
    win = np.hanning(frame)
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:n_frames]
    mag = np.abs(np.fft.rfft(frames * win, axis=1))
    logmag = np.log1p(100 * mag)
    flux = np.maximum(0.0, np.diff(logmag, axis=0)).sum(axis=1)
    flux = np.concatenate([[logmag[0].sum()], flux])
    if flux.max() <= 0:
        return []
    flux /= flux.max()

    # adaptive threshold: local moving average + delta
    k = max(1, int(0.1 * sr / hop))
    padded = np.pad(flux, k, mode="edge")
    local = np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), "valid")
    peaks, _ = find_peaks(flux, height=local + delta,
                          distance=max(1, int(min_gap * sr / hop)))
    # also allow a peak at frame 0 (sound starting right at the beginning)
    if len(flux) > 1 and flux[0] > flux[1] and flux[0] > local[0] + delta:
        peaks = np.concatenate([[0], peaks])
    # the flux peaks when the attack is roughly mid-frame
    times = np.maximum(0, peaks * hop + frame // 2) / sr
    times[peaks == 0] = 0.0
    return [float(t) for t in times]


def transcribe(x: np.ndarray, sr: int, settings: Settings | None = None,
               segment: bool = True, max_window: float = 2.0,
               skip_attack: float = 0.03, silence_db: float = -50.0):
    """Transcribe a recording.

    If ``segment`` is True the audio is split at detected onsets and each
    segment is analysed separately; otherwise the whole file is analysed as
    one chord. Returns ``(segments, a4)``.
    """
    settings = settings or Settings()
    duration = len(x) / sr
    if not segment:
        notes, a4 = detect_notes(x, sr, settings)
        return [Segment(0.0, duration, notes)], a4

    onsets = detect_onsets(x, sr)
    if not onsets:
        onsets = [0.0]
    # a single global tuning estimate is more robust than per-segment ones
    if settings.a4 is None:
        freqs, mag = magnitude_spectrum(x, sr)
        pf, pa = spectral_peaks(freqs, mag, floor_db=settings.peak_floor_db)
        a4 = estimate_tuning(pf, pa)
    else:
        a4 = settings.a4
    seg_settings = Settings(**{**settings.__dict__, "a4": a4})

    global_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    bounds = list(onsets) + [duration]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        a = int((start + skip_attack) * sr)
        b = int(min(end, start + max_window) * sr)
        if b - a < int(0.05 * sr):
            a = int(start * sr)
        chunk = x[a:b]
        if len(chunk) < 256:
            continue
        rms = np.sqrt(np.mean(chunk ** 2))
        if 20 * np.log10(rms / global_rms + 1e-12) < silence_db:
            continue
        notes, _ = detect_notes(chunk, sr, seg_settings)
        if notes:
            segments.append(Segment(start, end, notes))
    return segments, a4
