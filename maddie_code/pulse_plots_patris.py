import sys
import numpy as np
import scipy.signal as sgn
import librosa
import matplotlib.pyplot as plt

"""
Patris et al. 2019

Fourier transform of a pulsed sound shows peaks of frequencies, with a constant separation between frequencies... This band interval corresponds, in the time domain, to the repetition rate of pulses, or pulse rate,  called f_pulse in our study. These bands are often called side-bands in the literature

In order to better understand and characterize the pulsed sound, we will examine the relation between this side-band separation (pulse rate) and the abscissa of the peaks.

f_i = frequency value for the peaks in the spectrum

tonal means: 
there exists k st. f_i = k*delta_f
in this case, delta_f = f_pulse

otherwise: non-tonal

"""

sr = 20000  # Hz

## Model A 

#Sound parameters
d = 1.2  # sec
n_samples = int(d * sr)
f_pulse = 6  # Hz
T_pulse = 1/f_pulse  # sec
f_0 = 31.7  # Hz
sigma = 0.02  # sec

# Wavelet
t = np.arange(n_samples)/sr
t_wavelet = np.arange(-n_samples/2, n_samples/2)/sr
gaussian = np.exp(-0.5 * (t_wavelet / sigma) ** 2)
sine = np.sin(2*np.pi*f_0*t_wavelet)
# for i in range(2, 5):
#     harmonic_wavelet += np.sin(i*2*np.pi*f_0*t_wavelet)
wavelet = sine * (gaussian / np.max(gaussian))

# Dirac comb
dirac_comb = np.zeros(n_samples)
for i in np.arange(int(4*sigma*sr), n_samples, int(T_pulse * sr)):
    dirac_comb[i] = 1.

# Full signal
s = np.convolve(wavelet, dirac_comb, mode="same")

# Time domain plot
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(24,8))
axs[0].plot(t, wavelet)
axs[1].plot(t, dirac_comb)
axs[2].plot(np.arange(len(s))/sr, s)
plt.savefig("./outputs//modela.png")
plt.close()

# Frequency domain plot
for k, _s in {"signal": s, "dirac": dirac_comb, "wavelet": wavelet}.items():
    yf = np.fft.fftshift(np.fft.fft(_s))
    xf = np.fft.fftshift(np.fft.fftfreq( len(_s), 1/sr))
    yf = yf[int(len(yf)/2):int(len(yf)/2)+500]
    xf = xf[int(len(xf)/2):int(len(xf)/2)+500]
    plt.plot(xf, np.abs(yf), label=k, alpha=0.7)

for i in range(10):
    plt.axvline(i * f_pulse, alpha=0.2)

plt.axvline(f_0, alpha=0.2, color="red")
plt.legend()
plt.savefig("./outputs//modela_fft.png")
plt.close()

## Model B 

for sound_type, f_0 in {"tonal": 30, "non_tonal": 31.7}.items():
    g = np.sin(2*np.pi*f_0*t)
    pulse = np.convolve(gaussian/np.max(gaussian), dirac_comb, mode="same")
    s = g * pulse

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(24,8))
    axs[0].plot(t, g)
    axs[1].plot(t, pulse)
    axs[2].plot(t, s)
    plt.savefig(f"./outputs//modelb_{sound_type}.png")
    plt.close()

    for k, _s in {"signal": s, "pulse": pulse, "g": g}.items():
        yf = np.fft.fftshift(np.fft.fft(_s))
        xf = np.fft.fftshift(np.fft.fftfreq( len(_s), 1/sr))
        yf = yf[int(len(yf)/2):int(len(yf)/2)+100]
        xf = xf[int(len(xf)/2):int(len(xf)/2)+100]
        plt.plot(xf, np.abs(yf), label=k)

    for i in range(10):
        plt.axvline(i * f_pulse, alpha=0.2)

    plt.axvline(f_0, alpha=0.2, color="red")
    plt.legend()

    plt.savefig(f"./outputs/modelb_fft_{sound_type}.png")
    plt.close()

