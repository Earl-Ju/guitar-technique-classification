# first_audio_analysis.py
from __future__ import annotations
import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

# -------- Config --------
SR = 22050          # sample rate
DURATION = 3.0      # seconds
FREQ = 440.0        # Hz (A4)
AMP = 0.5
OUT_DIR = "results/quick_test"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def gen_sine(freq: float = FREQ, sr: int = SR, duration: float = DURATION, amp: float = AMP) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def plot_and_save_waveform(y: np.ndarray, sr: int, path: str):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_and_save_log_spectrogram(y: np.ndarray, sr: int, path: str):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, y_axis="log", x_axis="time", hop_length=512)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Frequency Spectrogram (STFT)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_and_save_melspectrogram(y: np.ndarray, sr: int, path: str):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=sr//2)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, y_axis="mel", x_axis="time", hop_length=512)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-Spectrogram")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_and_save_chroma(y: np.ndarray, sr: int, path: str):
    # 对纯正弦来说 chroma 会是一条主要能量所在的音名列
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(12, 3.5))
    librosa.display.specshow(C, x_axis="time", y_axis="chroma", sr=sr)
    plt.colorbar()
    plt.title("Chroma (CQT)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    ensure_dir(OUT_DIR)

    # 1) 合成音频
    y = gen_sine()

    # 2) 保存 WAV（float32）
    wav_path = os.path.join(OUT_DIR, "tone_440hz.wav")
    sf.write(wav_path, y, SR, subtype="PCM_16")

    # 3) 作图并保存
    plot_and_save_waveform(y, SR, os.path.join(OUT_DIR, "waveform.png"))
    plot_and_save_log_spectrogram(y, SR, os.path.join(OUT_DIR, "spectrogram_log.png"))
    plot_and_save_melspectrogram(y, SR, os.path.join(OUT_DIR, "melspectrogram.png"))
    plot_and_save_chroma(y, SR, os.path.join(OUT_DIR, "chroma.png"))

    # 4) 合并展示一张总览（可视检查）
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    # 波形
    librosa.display.waveshow(y, sr=SR, ax=axs[0, 0])
    axs[0, 0].set_title("Waveform")
    # STFT 对数频谱
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(D, sr=SR, y_axis="log", x_axis="time", hop_length=512, ax=axs[0, 1])
    fig.colorbar(img, ax=axs[0, 1], format="%+2.0f dB")
    axs[0, 1].set_title("Log-Frequency Spectrogram")
    # Mel
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=SR//2)
    S_db = librosa.power_to_db(S, ref=np.max)
    img2 = librosa.display.specshow(S_db, sr=SR, y_axis="mel", x_axis="time", hop_length=512, ax=axs[1, 0])
    fig.colorbar(img2, ax=axs[1, 0], format="%+2.0f dB")
    axs[1, 0].set_title("Mel-Spectrogram")
    # Chroma
    C = librosa.feature.chroma_cqt(y=y, sr=SR)
    img3 = librosa.display.specshow(C, x_axis="time", y_axis="chroma", sr=SR, ax=axs[1, 1])
    fig.colorbar(img3, ax=axs[1, 1])
    axs[1, 1].set_title("Chroma (CQT)")
    plt.tight_layout()
    plt.show()

    print("\n✅ Quick test done. Artifacts:")
    for f in ["tone_440hz.wav", "waveform.png", "spectrogram_log.png", "melspectrogram.png", "chroma.png"]:
        print("  -", os.path.join(OUT_DIR, f))

if __name__ == "__main__":
    main()
