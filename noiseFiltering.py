import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy.signal import butter, lfilter

#import rekaman .wav
audio = "rekaman.wav"

if not os.path.exists(audio):
    raise FileNotFoundError(f"File '{audio}' tidak ditemukan dalam folder ini")

#membaca file .wav
fs, data = wavfile.read(audio)
print(f"File '{audio}' berhasil diimpor. Sampling rate: {fs}, Shape: {data.shape}")

#plotting sinyal rekaman asli
t = np.linspace(0, len(data) / fs, num=len(data))
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, data, color='blue')
plt.title("Sinyal Rekaman Asli")
plt.xlabel("Waktu (detik)")
plt.ylabel("Amplitudo")

#menambahkan white noise
noise = np.random.normal(0,500, data.shape)
data_noisy = data + noise.astype(np.int16)

#menyimpan rekaman noisy
wavfile.write('rekaman_dengan_noise.wav', fs, data_noisy)

#plotting sinyal rekaman setelah ditambahkan noise
plt.subplot(3,1,2)
plt.plot(t, data_noisy, color='red')
plt.title("Sinyal Rekaman Noisy")
plt.xlabel("Waktu (detik)")
plt.ylabel("Amplitudo")

#menghilangkan noise dengan low-pass filter
#buat function filter low pass
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

#menerapkan filter pada rekaman noisy
cutoff = 4000
filtered_data = butter_lowpass_filter(data_noisy, cutoff, fs)

#menyimpan hasil denoising
wavfile.write('sura_denoised.wav', fs, filtered_data.astype(np.int16))

#memvisualisasikan dengan matplotlib
plt.subplot(3, 1, 3)
plt.plot(t, filtered_data, color='green')
plt.title("Sinyal Setelah Denoising Low-Pass Filter")
plt.xlabel("Waktu (detik)")
plt.ylabel("Amplitudo")

plt.tight_layout()
plt.show()