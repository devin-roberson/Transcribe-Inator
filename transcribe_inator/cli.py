"""Command-line interface: ``transcribe-inator song.wav``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .analysis import Settings, magnitude_spectrum, transcribe
from .audio import load_wav
from .notes import name_to_midi
from .synth import chord, sequence


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="transcribe-inator",
        description="Analyse a .wav file and report the musical notes being played.",
    )
    p.add_argument("wav", nargs="?", help="input .wav file")
    p.add_argument("--whole", action="store_true",
                   help="treat the whole file as a single chord (no onset segmentation)")
    p.add_argument("--a4", type=float, default=None,
                   help="tuning reference in Hz (default: estimate from the audio)")
    p.add_argument("--max-notes", type=int, default=8,
                   help="maximum simultaneous notes per chord (default 8)")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="minimum note strength relative to the strongest note, 0-1 "
                        "(lower finds quieter notes but more false positives; default 0.2)")
    p.add_argument("--lowest", default="A0", help="lowest note to consider (default A0)")
    p.add_argument("--highest", default="C8", help="highest note to consider (default C8)")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("--plot", metavar="PNG",
                   help="save a spectrum plot with detected notes marked (needs matplotlib)")
    p.add_argument("--demo", metavar="OUT_WAV",
                   help="write a synthesized demo .wav (C, F, G7, C chords) and exit")
    return p.parse_args(argv)


def _write_demo(path: str) -> None:
    from scipy.io import wavfile
    sr = 44100
    x = sequence([["C3", "E4", "G4"], ["F3", "A3", "C4"],
                  ["G2", "B3", "D4", "F4"], ["C3", "G3", "E4"]], 1.2, sr)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))


def _plot(path, x, sr, segments):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("--plot needs matplotlib (pip install matplotlib)", file=sys.stderr)
        return
    n = len(segments)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), squeeze=False)
    for ax, seg in zip(axes[:, 0], segments):
        chunk = x[int(seg.start * sr):int(min(seg.end, seg.start + 2.0) * sr)]
        freqs, mag = magnitude_spectrum(chunk, sr)
        mask = (freqs > 20) & (freqs < 5000)
        ax.semilogx(freqs[mask], 20 * np.log10(mag[mask] + 1e-9), lw=0.7, color="#555")
        for note in seg.notes:
            ax.axvline(note.frequency, color="#d62728", lw=1, alpha=0.8)
            ax.text(note.frequency, ax.get_ylim()[1], note.name, color="#d62728",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("dB")
        ax.set_title(f"{seg.start:.2f}s - {seg.end:.2f}s", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved spectrum plot to {path}", file=sys.stderr)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.demo:
        _write_demo(args.demo)
        print(f"Wrote demo audio to {args.demo}")
        return 0
    if not args.wav:
        print("error: an input .wav file is required (or use --demo OUT.wav)", file=sys.stderr)
        return 2

    try:
        x, sr = load_wav(args.wav)
    except (OSError, ValueError) as e:
        print(f"error: could not read {args.wav}: {e}", file=sys.stderr)
        return 1
    if len(x) == 0:
        print("error: the file contains no audio", file=sys.stderr)
        return 1

    settings = Settings(a4=args.a4, max_notes=args.max_notes, threshold=args.threshold,
                        low_note=name_to_midi(args.lowest),
                        high_note=name_to_midi(args.highest))
    segments, a4 = transcribe(x, sr, settings, segment=not args.whole)

    if args.json:
        print(json.dumps({"file": args.wav, "sample_rate": sr,
                          "duration": round(len(x) / sr, 3), "a4": round(a4, 2),
                          "segments": [s.to_dict() for s in segments]}, indent=2))
    else:
        print(f"{args.wav}: {len(x) / sr:.2f}s @ {sr} Hz, tuning A4 = {a4:.1f} Hz")
        if not segments:
            print("No notes detected.")
        for seg in segments:
            names = " ".join(n.name for n in seg.notes)
            print(f"\n[{seg.start:6.2f}s - {seg.end:6.2f}s]  {names}")
            for n in seg.notes:
                print(f"    {n.name:<4} {n.frequency:8.2f} Hz  {n.cents_off:+5.0f} cents"
                      f"  strength {n.strength_db:6.1f} dB")

    if args.plot and segments:
        _plot(args.plot, x, sr, segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
