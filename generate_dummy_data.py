# generate_dummy_data.py
import os
import numpy as np
import soundfile as sf

# 采样率
sr = 22050
duration = 2.0  # 每个样本 2 秒

def save_wave(path, signal):
    sf.write(path, signal, sr)

# 输出目录
base_dir = "data/processed"
os.makedirs(os.path.join(base_dir, "bending"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "vibrato"), exist_ok=True)

# 生成 bending（从 440Hz -> 660Hz 的线性上升音高）
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
bending_signal = np.sin(2 * np.pi * (440 + (220 * t / duration)) * t)
save_wave(os.path.join(base_dir, "bending", "bend1.wav"), bending_signal)
save_wave(os.path.join(base_dir, "bending", "bend2.wav"), bending_signal * 0.8)

# 生成 vibrato（440Hz 带 5Hz 抖动）
vibrato_signal = np.sin(2 * np.pi * (440 + 10 * np.sin(2 * np.pi * 5 * t)) * t)
save_wave(os.path.join(base_dir, "vibrato", "vib1.wav"), vibrato_signal)
save_wave(os.path.join(base_dir, "vibrato", "vib2.wav"), vibrato_signal * 0.8)

print("✅ Dummy dataset generated at data/processed/")
