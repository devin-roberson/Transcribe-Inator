"""WAV loading."""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a .wav file as mono float64 samples in [-1, 1].

    Returns ``(samples, sample_rate)``. Supports 8/16/24/32-bit integer PCM
    and 32/64-bit float files; multi-channel audio is averaged to mono.
    """
    sr, data = wavfile.read(path)
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)
