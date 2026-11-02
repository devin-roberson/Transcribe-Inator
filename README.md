# Transcribe-Inator

Give it a `.wav` file and it works out which musical notes are being played,
including several notes at once, like the notes of a piano chord.

```
$ transcribe-inator chords.wav
chords.wav: 4.80s @ 44100 Hz, tuning A4 = 440.1 Hz

[  0.00s -   1.20s]  C3 E4 G4
    C3     130.83 Hz     -0 cents  strength    0.0 dB
    E4     329.68 Hz     -0 cents  strength   -1.9 dB
    G4     392.13 Hz     -0 cents  strength   -4.0 dB

[  1.20s -   2.39s]  F3 A3 C4
...
```

## Install

Requires Python 3.9+.

```bash
pip install -e .            # numpy + scipy
pip install -e ".[plot]"    # optional: matplotlib, for --plot
```

## Usage

```bash
transcribe-inator song.wav                 # split at note attacks, list the notes in each chord
transcribe-inator chord.wav --whole        # treat the whole file as one chord
transcribe-inator song.wav --json          # machine-readable output
transcribe-inator song.wav --plot out.png  # spectrum of each segment with the detected notes marked
transcribe-inator --demo demo.wav          # write a synthesized test file to try it on
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--a4 HZ` | Tuning reference. By default it's estimated from the audio, so slightly sharp or flat instruments still work. |
| `--threshold 0-1` | Minimum note strength relative to the loudest note (default 0.2). Lower it to catch quiet notes, raise it if you see false notes. |
| `--max-notes N` | Maximum number of simultaneous notes (default 8). |
| `--lowest / --highest` | Note range to search, e.g. `--lowest C2 --highest C7` (default is the full piano, A0 to C8). |

It also works as a library:

```python
from transcribe_inator import load_wav, transcribe, detect_notes

x, sr = load_wav("chord.wav")
notes, a4 = detect_notes(x, sr)
print([n.name for n in notes])             # ['C4', 'E4', 'G4']

segments, a4 = transcribe(x, sr)           # one list of notes per detected attack
```

## How it works

A note played on a piano isn't a single frequency. It's a fundamental `f0`
plus overtones at roughly `2·f0, 3·f0, 4·f0, …`. In a chord, the overtones of
different notes overlap, so the loudest peaks can't simply be read off as
notes. The pipeline (`transcribe_inator/analysis.py`) goes like this:

1. **Onset detection**: spectral flux finds the moment each new chord or note
   is struck, and the audio is split into segments at those points.
2. **Spectrum**: a Hann-windowed, zero-padded FFT of each segment. Spectral
   peaks are picked above an adaptive noise floor and refined to sub-bin
   accuracy with parabolic interpolation.
3. **Tuning estimate**: an amplitude-weighted circular mean of the strongest
   peaks' deviation from equal temperament.
4. **Harmonic salience**: each candidate note from A0 to C8 is scored by the
   weighted sum of the peaks at its harmonic positions. Partial positions are
   tracked as they go up, so the slightly sharp overtones of real piano
   strings (inharmonicity) still match. A candidate must have its fundamental
   present, which stops a note from being "found" an octave or a fifth below
   a note that was actually played.
5. **Iterative estimation and cancellation**: the best candidate is accepted
   and a smooth spectral envelope fitted to its partials is subtracted. That
   way, partials it shares with other notes keep their excess for those
   notes. This repeats until what's left is too weak.

## Accuracy and limitations

On synthesized piano-like tones (with inharmonicity, decaying partials and
noise), the test suite gets every note right for triads, 7th chords, 5-note
chords, low bass voicings and high-register chords. On 50 random 1–5 note
chords, precision and recall are both about 99%.

- **Octave doublings are the hardest case.** If C3 and C4 sound together,
  every partial of C4 lands on a partial of C3. Simple cases work (see the
  tests), but in dense voicings with doubled octaves the upper octave note is
  sometimes missed (about 85% recall on octave-heavy voicings).
- Very quiet notes under loud ones can fall below `--threshold`.
- The analysis looks at the start (up to 2 s) of each segment between
  attacks. Notes held over from an earlier segment are reported again if they
  are still sounding.
- It is tuned for pitched, harmonic instruments (piano, guitar, organ,
  voice). Drums and heavy distortion will confuse it.

## Development

```bash
pip install -e ".[dev]"
pytest
```

`transcribe_inator/synth.py` generates the piano-like test tones.
