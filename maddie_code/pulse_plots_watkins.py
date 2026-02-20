import sys
import numpy as np
import scipy.signal as sgn
import librosa
import matplotlib.pyplot as plt

"""
Watkins 1968

Pulse trains not discussed properly, this paper meant to help
- Often, frequencies are not harmonically related
- Unequal emphasis on "harmonics" is present 
- Fundamental is "missing"

"""

def fft_plot(_s, sr):
    yf = np.fft.fftshift(np.fft.fft(_s))
    xf = np.fft.fftshift(np.fft.fftfreq( len(_s), 1/sr))
    yf = yf[int(len(yf)/2):]
    xf = xf[int(len(xf)/2):]
    return xf, np.abs(yf)

sr = 20000

# Simple 1000-Hz sine-wave
# No pulse, so no side-bands
# I think: this is what people call "whistle"
# As pulses get longer, they overlap <-> whistle

d = 3
t = np.arange(int(3*sr))/sr
f_0 = 1000
w = np.sin(2*np.pi*f_0*t)

fig, axs = plt.subplots(2,1)
axs[0].plot(t, w)
xf, yf = fft_plot(w, sr)
axs[1].plot(xf, yf)
axs[1].axvline(f_0, alpha=0.6, color="red")
plt.savefig("./outputs/watkins/watkins_0.png")
plt.close()

# Fig1A: 1000-Hz tone with 166 or 500 Hz cycle
# Extra frequencies are at "beat frequencies" or "sidebands", i.e.,
# f_0 plus/minus integer multiples of f_pulse
fig, axs = plt.subplots(2,2, sharey='col', sharex='col')
for i, (n_cycles, f_pulse) in enumerate([(2, 166), (1, 500)]):

	f_0 = 1000
	d_cycles = n_cycles * (1/f_0)
	t_sine = np.arange(int(d_cycles*sr))/sr
	w = np.sin(2*np.pi*f_0*t_sine)

	T_pulse = 1/f_pulse
	r = np.zeros(int(sr*T_pulse))
	r[len(w):2*len(w)] = w

	if n_cycles == 2:
		s = np.concatenate([r, r])
	else:
		s = np.concatenate([r, r, r, r, r])
	t = np.arange(len(s))/sr

	axs[i, 0].plot(t, s)
	xf, yf = fft_plot(s, sr)
	axs[i, 1].plot(xf, yf)
	axs[i, 1].axvline(f_0, alpha=0.6, color="red")
	for j in range(10):
		axs[i, 1].axvline(j*f_pulse, alpha=0.3, color="green")
	axs[i, 1].set_xlim([0, 2000])

plt.savefig("./outputs/watkins/watkins_fig1.png")
plt.close()


# Fig2A-E: 1000-Hz tone with varied cycles
# Keep f_0 of 1000 Hz with two pulses, but f_pulse is varied
# We see change in beat frequencies as before,
# Relative amplitude of f_0 relative to sidebands
# Pulse duration relative to "interval between pulses" ("duty cycle")
# controls amount of energy in pulse tone-frequency
# In this example, there's a reduction in intensity at 1000 hz
fig, axs = plt.subplots(4, 2, sharey='col', sharex='col')
cycles = [(2, 2), (2,4), (2,8), (2,16)]
for i, (on_cycle, off_cycle) in enumerate(cycles):

	f_0 = 1000
	d_cycles_on = on_cycle * (1/f_0)
	t_sine = np.arange(int(d_cycles_on*sr))/sr
	on_w = np.sin(2*np.pi*f_0*t_sine)

	d_cycles_off = off_cycle * (1/f_0)
	off_w = np.zeros(int(d_cycles_off*sr))
	f_pulse = 1/(d_cycles_on + d_cycles_off)

	w = np.concatenate((on_w, off_w))
	s = np.concatenate([w, w, w, w])
	t = np.arange(len(s))/sr

	axs[i, 0].plot(t, s)
	xf, yf = fft_plot(s, sr)
	axs[i, 1].plot(xf, yf)
	axs[i, 1].axvline(f_0, alpha=0.6, color="red")
	for j in range(10):
		axs[i, 1].axvline(j*f_pulse, alpha=0.3, color="green")
	axs[i, 1].set_xlim([0, 2000])

plt.savefig("./outputs/watkins/watkins_fig2.png")
plt.close()


# Fig3: 1000-Hz tone with varied cycles
# f_pulse is the same (250 Hz) in all of these examples
# But, the energy distribution across sidebands varies
# In particular, symmetry of individual pulses and
# the silences creates absence of even-numbered sidebands
# ==> This is the lowest number of sidebands
# More sidebands when pulse duration is either very long or very short
# relative to interval between pulses
# As in Fig2: Pulse duration relative to "interval between pulses" ("duty cycle")
# controls amount of energy in pulse tone-frequency
# In this example, there's a increase in intensity at 1000 hz
fig, axs = plt.subplots(3, 2, sharey='col', sharex='col')
cycles = [(1, 3), (2,2), (3,1)]
for i, (on_cycle, off_cycle) in enumerate(cycles):

	f_0 = 1000
	d_cycles_on = on_cycle * (1/f_0)
	t_sine = np.arange(int(d_cycles_on*sr))/sr
	on_w = np.sin(2*np.pi*f_0*t_sine)

	d_cycles_off = off_cycle * (1/f_0)
	off_w = np.zeros(int(d_cycles_off*sr))
	f_pulse = 1/(d_cycles_on + d_cycles_off)

	w = np.concatenate((on_w, off_w))
	s = np.concatenate([w, w, w, w])
	t = np.arange(len(s))/sr

	axs[i, 0].plot(t, s)
	xf, yf = fft_plot(s, sr)
	axs[i, 1].plot(xf, yf)
	axs[i, 1].axvline(f_0, alpha=0.6, color="red")
	for j in range(10):
		axs[i, 1].axvline(j*f_pulse, alpha=0.3, color="green")
	axs[i, 1].set_xlim([0, 2000])

plt.savefig("./outputs/watkins/watkins_fig3.png")
plt.close()


# Fig 7 - looking now at spikes
# There are two extremes in pulses: sine-wave with a single frequency characteristic
# the spike ( a very short pulse of energy ), as if composed of all frequencies
# Other pulse times appear to fall somewhere in between
# Rapid train of spikes resolves into discrete harmonic bands
# Interval between harmonics indicates the repetition rate (f_pulse), but in contrast to 
# sine-wave pulse trains, there is considerable energy at f_pulse Hz

d = 2.0
t = np.arange(0, d, 1/sr)
f = np.zeros_like(t)

# Segment 1
mask1 = t < 0.5
f[mask1] = 50

# Segment 2
mask2 = (t>=0.5) & (t<1.0)
f[mask2] = 50 + (t[mask2] - 0.5) * (400-50)/0.5

# Segment 3: 1.0 to 1.5s (400 Hz)
mask3 = (t>=1.0) & (t<1.5)
f[mask3] = 400

# Segment 4: 1.5 to 2.0 (Ramp down 400 Hz to 50 hz)
mask4 = (t > 1.5) & (t<2.0)
f[mask4] = 400 - (t[mask4] - 1.5)* (400-50)/0.5

# Segment 5: 2.0 to end (50 Hz)
mask5 = t >= 2.0
f[mask5] = 50

# Generate spike train
# Integrate the frequency to get the phase (angle)
# phase = 2 * pi * integral(f(t) / fs)
phase = 2 * np.pi * np.cumsum(f / sr)
# Threshold the sinewave to get descrete spikes
spikes = np.sin(phase) > 0.99

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(t, 1.0*spikes)
S = librosa.stft(y=1.0*spikes, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
S_dB = librosa.power_to_db(np.abs(S), ref=np.max(np.abs(S)))
img = librosa.display.specshow(
	S_dB, x_axis='time',y_axis='linear',
	n_fft=n_fft, win_length=win_length, hop_length=hop_length,
	sr=sr,ax=axs[1], vmin=-20, cmap="viridis")
axs[1].set_ylim([0, 2000])
plt.savefig("./outputs/watkins/watkins_fig7.png")
plt.close()

# Figure 8
# Train of spikes may be pulsed in bursts
# Similar to pulsing sine waves
# Relative strength at f_spike increases with repetitions
# Compared to energy at sidebands
fig, axs = plt.subplots(3, 2, sharex='col')
for n_pulses in [1, 2, 3]:
	# Note that these look like harmonics of 250
	# But if we used something like 166 Hz it would
	# be obvious that they were sidebands.
	f_pulse = 250
	T_pulse = 1/f_pulse
	t_pulse = np.zeros(int(T_pulse * sr))

	f_spike = 1000
	T_spike = 1/1000
	for pulse_idx in range(n_pulses):
		spike_idx = int(pulse_idx * T_spike * sr)
		t_pulse[spike_idx] = 1

	s = np.tile(t_pulse, 100)
	t = np.arange(0, len(s)) / sr

	axs[n_pulses - 1, 0].plot(t[:1000], s[:1000])
	xf, yf = fft_plot(s, sr)
	axs[n_pulses - 1, 1].plot(xf, yf)
	axs[n_pulses - 1, 1].axvline(f_spike, alpha=0.6, color="red")
	for j in range(10):
		axs[n_pulses - 1, 1].axvline(j*f_pulse, alpha=0.3, color="green")
	# axs[i, 1].set_xlim([0, 2000])

plt.savefig("./outputs/watkins/watkins_fig8.png")
plt.close()


# 


