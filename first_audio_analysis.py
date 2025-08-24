# first_audio_analysis.py
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. 生成一段简单的测试音频（一个440Hz的正弦波，即A4音符）
sample_rate = 22050  # 采样率
duration = 3.0       # 持续时间3秒
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 440 Hz sine wave

# 2. 计算并绘制波形图
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
librosa.display.waveshow(audio_signal, sr=sample_rate)
plt.title('Waveform of Generated 440 Hz Tone')

# 3. 计算并绘制频谱图（Spectrogram）
plt.subplot(2, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal)), ref=np.max)
librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sample_rate)
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Frequency Spectrogram')

plt.tight_layout()
plt.show()

# 4. 打印一些基本音频信息
print("Audio signal generated successfully!")
print(f"Signal shape: {audio_signal.shape}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {duration} seconds")