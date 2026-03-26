"""
Utilities for audio processing and visualization in the contour workflow.

- pitch_shift: Shift pitch using librosa.effects.pitch_shift
- plot_spectrogram: Plot spectrogram from audio array (y, sr)
- plot_spectrogram_from_file: Plot spectrogram from file with optional pitch overlay
"""
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pitch_shift(
    y: np.ndarray,
    sr: int,
    n_steps: float,
    *,
    bins_per_octave: int = 12,
    res_type: str = "soxr_hq",
    scale: bool = True,
) -> np.ndarray:
    """
    Shift the pitch of an audio waveform.

    Uses librosa.effects.pitch_shift. Positive n_steps shifts up, negative shifts down.
    One step = one semitone when bins_per_octave=12.

    Args:
        y: Audio time series (mono or stereo).
        sr: Sample rate.
        n_steps: Number of (fractional) semitones to shift.
        bins_per_octave: Steps per octave (default 12 = semitones).
        res_type: Resampling algorithm.
        scale: Whether to scale output to match input energy.

    Returns:
        Pitch-shifted audio array.

    Example:
        >>> y, sr = librosa.load("audio.mp3")
        >>> y_up = pitch_shift(y, sr, n_steps=2)   # up 2 semitones
        >>> y_down = pitch_shift(y, sr, n_steps=-3)  # down 3 semitones
    """
    return librosa.effects.pitch_shift(
        y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave,
        res_type=res_type, scale=scale
    )


def plot_spectrogram(
    y: np.ndarray,
    sr: int,
    *,
    pitch_time: np.ndarray | None = None,
    pitch_freq: np.ndarray | None = None,
    title: str | None = None,
    n_fft: int = 2048,
    hop_length: int = 256,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """
    Plot linear spectrogram from audio array.

    Use this for in-memory audio (e.g. pitch-shifted) without loading from file.

    Args:
        y: Audio time series.
        sr: Sample rate.
        pitch_time: Optional time array (seconds) for pitch contour overlay.
        pitch_freq: Optional frequency array (Hz) for pitch contour overlay.
        title: Optional plot title.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        figsize: Figure size.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    fig, ax = plt.subplots(figsize=figsize)

    img = librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax,
    )

    if pitch_time is not None and pitch_freq is not None:
        ax.plot(pitch_time, pitch_freq, color="cyan", linewidth=2, label="Pitch contour")
        ax.legend(loc="upper right")
        ax.set_title(title or "Spectrogram + pitch contour")
    else:
        ax.set_title(title or "Spectrogram")

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    plt.show()


def plot_spectrogram_from_file(
    folder_path: str,
    filename: str,
    *,
    plot_pitch: bool = True,
    n_fft: int = 2048,
    hop_length: int = 256,
    figsize: tuple[float, float] = (12, 5),
) -> None:
    """
    Plot linear spectrogram from file, optionally with pitch contour from .f0.csv.

    Args:
        folder_path: Path to folder containing audio and (optionally) .f0.csv.
        filename: Base filename without extension (e.g. 'N01i-A1-1').
            If extension is present (e.g. 'N01i-A1-1.mp3'), it is stripped.
        plot_pitch: If True, load {filename}.f0.csv and overlay pitch contour.
            If False, plot spectrogram only.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        figsize: Figure size.
    """
    base = Path(filename).stem
    folder = os.path.abspath(folder_path)

    # Resolve audio path (support .mp3 and .wav)
    audio_path = None
    for ext in (".mp3", ".wav"):
        p = os.path.join(folder, f"{base}{ext}")
        if os.path.exists(p):
            audio_path = p
            break
    if audio_path is None:
        raise FileNotFoundError(f"No audio file {base}.mp3 or {base}.wav in {folder}")

    y, sr = librosa.load(audio_path)

    pitch_time = None
    pitch_freq = None
    if plot_pitch:
        f0_path = os.path.join(folder, f"{base}.f0.csv")
        if not os.path.exists(f0_path):
            raise FileNotFoundError(f"No pitch file {f0_path}")
        df = pd.read_csv(f0_path)
        pitch_time = df["time"].values / 1000.0
        pitch_freq = df["frequency"].values

    plot_spectrogram(
        y,
        sr,
        pitch_time=pitch_time,
        pitch_freq=pitch_freq,
        title=f"Spectrogram{' + pitch contour' if plot_pitch else ''}: {os.path.basename(audio_path)}",
        n_fft=n_fft,
        hop_length=hop_length,
        figsize=figsize,
    )
