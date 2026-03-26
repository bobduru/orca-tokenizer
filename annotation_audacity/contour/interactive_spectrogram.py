"""
Interactive spectrogram with sliders for n_fft and hop_length.

Usage:
    uv run python annotation_audacity/contour/interactive_spectrogram.py --file data/ford-catalogue/Northern\ Resident/media/A_clan/N07i-A1-1.mp3
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import librosa
import librosa.display


def run_interactive(audio_path: str, n_fft_init: int = 1024, hop_init: int = 256):
    y, sr = librosa.load(audio_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial spectrogram (specshow returns QuadMesh)
    def draw_spectrogram(n_fft, hop):
        win = min(n_fft, len(y))
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        return librosa.display.specshow(
            S_db, sr=sr, hop_length=hop, x_axis="time", y_axis="linear",
            ax=ax, n_fft=n_fft
        )

    img_ref = [draw_spectrogram(n_fft_init, hop_init)]
    ax.set_title(f"Spectrogram: {audio_path}")
    cbar = fig.colorbar(img_ref[0], ax=ax, format="%+2.0f dB")

    # Sliders: n_fft (powers of 2), hop_length
    ax_nfft = plt.axes([0.15, 0.12, 0.65, 0.03])
    ax_hop = plt.axes([0.15, 0.06, 0.65, 0.03])

    n_fft_min, n_fft_max = 256, 8192
    hop_min, hop_max = 64, 1024

    slider_nfft = Slider(ax_nfft, "n_fft", n_fft_min, n_fft_max, valinit=n_fft_init, valstep=256)
    slider_hop = Slider(ax_hop, "hop_length", hop_min, hop_max, valinit=hop_init, valstep=64)

    def update(_):
        n_fft = int(slider_nfft.val)
        hop = int(slider_hop.val)
        img_ref[0].remove()
        img_ref[0] = draw_spectrogram(n_fft, hop)
        cbar.update_normal(img_ref[0])
        fig.canvas.draw_idle()

    slider_nfft.on_changed(update)
    slider_hop.on_changed(update)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive spectrogram with n_fft and hop_length sliders")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to audio file (.mp3 or .wav)")
    parser.add_argument("--nfft", type=int, default=1024, help="Initial n_fft")
    parser.add_argument("--hop", type=int, default=256, help="Initial hop_length")
    args = parser.parse_args()
    run_interactive(args.file, args.nfft, args.hop)
