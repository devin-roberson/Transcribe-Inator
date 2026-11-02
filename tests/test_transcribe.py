import json

import numpy as np
import pytest
from scipy.io import wavfile

from transcribe_inator import Settings, detect_notes, detect_onsets, load_wav, transcribe
from transcribe_inator.cli import main
from transcribe_inator.notes import midi_to_name, name_to_midi
from transcribe_inator.synth import chord, piano_tone, sequence

SR = 44100


def names(notes):
    return [n.name for n in notes]


def test_note_name_roundtrip():
    assert name_to_midi("C4") == 60
    assert name_to_midi("A4") == 69
    assert name_to_midi("Bb3") == 58
    assert name_to_midi("F#5") == 78
    assert name_to_midi("A0") == 21
    assert midi_to_name(61) == "C#4"
    assert all(name_to_midi(midi_to_name(m)) == m for m in range(21, 109))


def test_pure_sine():
    t = np.arange(SR) / SR
    notes, _ = detect_notes(0.5 * np.sin(2 * np.pi * 440 * t), SR, Settings(a4=440.0))
    assert names(notes) == ["A4"]
    assert notes[0].frequency == pytest.approx(440.0, abs=0.5)


@pytest.mark.parametrize("notes", [
    ["A3"],
    ["C4", "E4", "G4"],
    ["C4", "D4"],                       # adjacent whole tone
    ["D4", "F#4", "A4", "C5"],          # dominant 7th
    ["B3", "D4", "F4", "G#4"],          # diminished 7th
    ["F4", "A4", "C5", "E5", "G5"],     # 5-note chord
    ["A1", "E2", "A2"],                 # low register
    ["C2", "G2", "E3", "C4"],           # spread voicing
    ["C3", "G3", "C4", "E4"],           # with an octave doubling
    ["C6", "E6", "G6"],                 # high register
])
def test_piano_chords(notes):
    detected, _ = detect_notes(chord(notes, 2.0, SR), SR)
    expected = sorted((midi_to_name(name_to_midi(n)) for n in notes), key=name_to_midi)
    assert names(detected) == expected


def test_auto_tuning_detects_detuned_piano():
    x = chord(["C4", "E4", "G4"], 2.0, SR, a4=446.0)
    detected, a4 = detect_notes(x, SR)
    assert a4 == pytest.approx(446.0, abs=1.0)
    assert names(detected) == ["C4", "E4", "G4"]


def test_robust_to_noise():
    rng = np.random.default_rng(0)
    x = chord(["E3", "G#3", "B3"], 2.0, SR) + rng.normal(0, 0.005, 2 * SR)
    detected, _ = detect_notes(x, SR)
    assert names(detected) == ["E3", "G#3", "B3"]


def test_onsets_and_segmentation():
    chords = [["C4", "E4", "G4"], ["F3", "A3", "C4"], ["G3", "B3", "D4"]]
    x = sequence(chords, 1.0, SR)
    onsets = detect_onsets(x, SR)
    assert len(onsets) == 3
    assert onsets == pytest.approx([0.0, 1.0, 2.0], abs=0.03)
    segments, _ = transcribe(x, SR)
    assert [names(s.notes) for s in segments] == chords


def test_silence_yields_nothing():
    segments, _ = transcribe(np.zeros(SR), SR)
    assert segments == []


def test_wav_loading_formats(tmp_path):
    x = chord(["A3", "C#4", "E4"], 1.0, SR)
    stereo = np.stack([x, x], axis=1)
    for dtype, scale in [(np.int16, 32767), (np.int32, 2**31 - 1), (np.float32, 1.0)]:
        path = tmp_path / f"t_{np.dtype(dtype).name}.wav"
        wavfile.write(path, SR, (stereo * scale).astype(dtype))
        y, sr = load_wav(str(path))
        assert sr == SR and y.ndim == 1
        assert np.max(np.abs(y - x)) < 1e-3


def test_cli_json(tmp_path, capsys):
    path = tmp_path / "chord.wav"
    wavfile.write(path, SR, (chord(["D4", "F4", "A4"], 1.5, SR) * 32767).astype(np.int16))
    assert main([str(path), "--whole", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [n["name"] for n in out["segments"][0]["notes"]] == ["D4", "F4", "A4"]


def test_cli_demo(tmp_path, capsys):
    path = tmp_path / "demo.wav"
    assert main(["--demo", str(path)]) == 0
    capsys.readouterr()
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "C3 E4 G4" in out and "G2 B3 D4 F4" in out
