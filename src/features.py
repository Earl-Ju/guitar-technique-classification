# src/features.py
from __future__ import annotations
import numpy as np
import librosa

def extract_features(
    file_path: str,
    sr: int = 22_050,
    n_mfcc: int = 20,
    include_chroma: bool = True,
    include_spectral: bool = True,
) -> np.ndarray:
    """Load audio and compute a compact feature vector."""
    y, _sr = librosa.load(file_path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio")

    feats = []

    # MFCC (mean + std)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats += [mfcc.mean(axis=1), mfcc.std(axis=1)]

    if include_chroma:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats += [chroma.mean(axis=1), chroma.std(axis=1)]

    if include_spectral:
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr       = librosa.feature.zero_crossing_rate(y)
        feats += [
            spec_cent.mean(axis=1), spec_cent.std(axis=1),
            spec_bw.mean(axis=1),   spec_bw.std(axis=1),
            rolloff.mean(axis=1),   rolloff.std(axis=1),
            zcr.mean(axis=1),       zcr.std(axis=1),
        ]

    return np.concatenate([f.ravel() for f in feats]).astype(np.float32)
