# src/features.py
from __future__ import annotations
import numpy as np
import librosa

def extract_features(
    file_path: str,
    sr: int = 22050,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    提取常用音频特征并拼成一个固定长度的一维向量（均值/标准差聚合）。

    含：MFCC, Chroma, Spectral Centroid, Bandwidth, Rolloff, ZCR
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # —— 频谱图基础
    S = np.abs(librosa.stft(y)) + 1e-9
    S_db = librosa.power_to_db(S**2, ref=np.max)

    # —— 各类特征（逐帧）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    # —— 统计汇总（均值 & 标准差）
    def stats(x: np.ndarray) -> np.ndarray:
        return np.hstack([x.mean(axis=1), x.std(axis=1)])

    feats = np.hstack([
        stats(mfcc),
        stats(chroma),
        stats(centroid),
        stats(bandwidth),
        stats(rolloff),
        stats(zcr),
    ])

    # 防 NaN/Inf
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)
