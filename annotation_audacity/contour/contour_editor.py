"""
Interactive contour editor: spectrogram with draggable pitch control points.

Usage:
    uv run annotation_audacity/contour/contour_editor.py --file N05i-A1-2 --harmonics 6
    uv run annotation_audacity/contour/contour_editor.py --file N16i-B1-1
    uv run annotation_audacity/contour/contour_editor.py --folder data/shifted_8
    uv run annotation_audacity/contour/contour_editor.py --file N18-B1-2 --save-folder annotations/
    uv run annotation_audacity/contour/contour_editor.py --file N16i-B1-1 --save-folder annotations/

Loading priority: saved CSV → audacity .txt → empty (0 parts).

No input prompts — always in editing mode.
- Drag dots to move. Click to select (highlighted orange).
- D: delete selected dot. A: add dot after selected (1000 Hz above).
- `: on a 2-dot part, expand to original f0 contour (~0.1 s spacing).
- Harmonics (2f0, 3f0, ...) shown as small colored dots.
"""
import os
import re
import json
import argparse
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox
import librosa
import librosa.display

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _find_parts_txt(folder: str, filename: str) -> str | None:
    folder = os.path.abspath(folder)
    candidates = [
        os.path.join(folder, f"{filename}.txt"),
        os.path.normpath(os.path.join(folder, "..", "..", "..", "audacity_maddie", "processed", f"{filename}.txt")),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def parse_old_audacity_parts(
    folder: str, filename: str
) -> tuple[list[tuple[str, float, float]], list[str]] | None:
    """Parse Audacity .txt label file for Px parts.

    Returns (parts, audacity_labels) where:
      parts          = [(simple_label, start_s, end_s), ...]  e.g. ("P1", 0.38, 0.57)
      audacity_labels = [original_full_label, ...]            e.g. "P1: BLUR"
    Returns None if no file found or no Px rows.
    """
    txt_path = _find_parts_txt(folder, filename)
    if txt_path is None:
        return None
    df = pd.read_csv(txt_path, sep="\t", header=None, names=["start", "end", "label"])
    parts, aud_labels = [], []
    px_re = re.compile(r"^P(\d+)\s*[:.]?", re.IGNORECASE)
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        m = px_re.match(label)
        if m:
            parts.append((f"P{m.group(1)}", float(row["start"]), float(row["end"])))
            aud_labels.append(label)
    if not parts:
        return None
    combined = sorted(zip(parts, aud_labels), key=lambda x: x[0][1])
    return [c[0] for c in combined], [c[1] for c in combined]


def load_audio_and_pitch(
    folder_path: str,
    filename: str | None = None,
    n_fft: int = 1024,
    hop_length: int = 128,
):
    folder = os.path.abspath(folder_path)
    if filename is None:
        for f in sorted(os.listdir(folder)):
            base, ext = os.path.splitext(f)
            if ext.lower() in (".mp3", ".wav") and os.path.exists(os.path.join(folder, f"{base}.f0.csv")):
                filename = base
                break
        if filename is None:
            raise FileNotFoundError(f"No audio + .f0.csv pair in {folder}")

    audio_path = None
    for ext in (".mp3", ".wav"):
        p = os.path.join(folder, f"{filename}{ext}")
        if os.path.exists(p):
            audio_path = p
            break
    if audio_path is None:
        raise FileNotFoundError(f"No {filename}.mp3/.wav in {folder}")

    f0_path = os.path.join(folder, f"{filename}.f0.csv")
    if not os.path.exists(f0_path):
        raise FileNotFoundError(f"No {f0_path}")

    y, sr = librosa.load(audio_path)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    df = pd.read_csv(f0_path)
    time_s = df["time"].values / 1000.0
    freq = df["frequency"].values
    return y, sr, S, time_s, freq, os.path.basename(audio_path)


_MIN_SPACING_S = 0.1  # minimum seconds between control points


def downsample_control_points(
    time_s: np.ndarray, freq: np.ndarray, n_points: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample to at most n_points with >= _MIN_SPACING_S between points; min 2."""
    n = len(time_s)
    if n <= 2:
        return time_s.copy(), freq.copy()
    duration = time_s[-1] - time_s[0]
    max_by_spacing = max(2, int(duration / _MIN_SPACING_S) + 1)
    actual_n = max(2, min(n_points, max_by_spacing))
    if n <= actual_n:
        return time_s.copy(), freq.copy()
    indices = np.linspace(0, n - 1, actual_n, dtype=int)
    return time_s[indices], freq[indices]


def get_control_points_by_parts(
    time_s: np.ndarray,
    freq: np.ndarray,
    parts: list[tuple[str, float, float]],
    n_points_per_part: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    all_t, all_f, part_indices = [], [], []
    for idx, (_, start, end) in enumerate(parts):
        mask = (time_s >= start) & (time_s <= end)
        t_part, f_part = time_s[mask], freq[mask]
        if len(t_part) == 0:
            continue
        t_ds, f_ds = downsample_control_points(t_part, f_part, n_points_per_part)
        all_t.extend(t_ds)
        all_f.extend(f_ds)
        part_indices.extend([idx] * len(t_ds))
    return np.array(all_t), np.array(all_f), part_indices


# ── persistence ───────────────────────────────────────────────────────────────

def load_saved_annotations(save_folder: str, filename: str) -> dict | None:
    """Load previously saved contour JSON.

    Returns a dict with keys:
      ctrl_time, ctrl_freq, part_indices, parts,
      parts_audacity_labels, parts_audacity_bounds,
      had_audacity_labels, parts_notes, general_notes
    or None if the file doesn't exist.
    """
    path = os.path.join(save_folder, f"{filename}_contour.json")
    if not os.path.exists(path):
        return None

    with open(path) as fh:
        data = json.load(fh)

    parts: list[tuple[str, float, float]] = []
    parts_audacity_labels: list[str] = []
    parts_audacity_bounds: list[tuple[float, float] | None] = []
    parts_notes: dict[int, str] = {}
    all_t, all_f, all_pi = [], [], []

    for pi, p in enumerate(data.get("parts", [])):
        pts = p.get("points", [])
        times = [pt["time_s"] for pt in pts]
        freqs = [pt["freq_hz"] for pt in pts]
        t_min = min(times) if times else 0.0
        t_max = max(times) if times else 0.0
        parts.append((p["label"], t_min, t_max))
        parts_notes[pi] = p.get("notes", "")

        aud = p.get("audacity_label")
        if aud:
            parts_audacity_labels.append(aud.get("note", ""))
            parts_audacity_bounds.append((aud["start"], aud["end"]))
        else:
            parts_audacity_labels.append("")
            parts_audacity_bounds.append(None)

        all_t.extend(times)
        all_f.extend(freqs)
        all_pi.extend([pi] * len(pts))

    for pt in data.get("unassigned_points", []):
        all_t.append(pt["time_s"])
        all_f.append(pt["freq_hz"])
        all_pi.append(-1)

    return {
        "ctrl_time": np.array(all_t),
        "ctrl_freq": np.array(all_f),
        "part_indices": np.array(all_pi, dtype=int),
        "parts": parts,
        "parts_audacity_labels": parts_audacity_labels,
        "parts_audacity_bounds": parts_audacity_bounds,
        "had_audacity_labels": data.get("had_audacity_labels", False),
        "raw_audacity_txt": data.get("audacity_label_txt", ""),
        "parts_notes": parts_notes,
        "general_notes": data.get("general_notes", ""),
    }


def save_annotations(
    save_folder: str,
    filename: str,
    editor,  # DraggableContour
    general_notes: str,
    parts_notes: dict[int, str],
) -> str:
    """Save contour dots + metadata to {save_folder}/{filename}_contour.json."""
    os.makedirs(save_folder, exist_ok=True)
    path = os.path.join(save_folder, f"{filename}_contour.json")

    # Collect dots per part
    part_points: dict[int, list[dict]] = {}
    unassigned: list[dict] = []
    for i, (t, f) in enumerate(zip(editor.time, editor.freq)):
        pi = int(editor.part_indices[i]) if editor.part_indices is not None else -1
        pt = {"time_s": round(float(t), 6), "freq_hz": round(float(f), 4)}
        if pi >= 0:
            part_points.setdefault(pi, []).append(pt)
        else:
            unassigned.append(pt)

    parts_out = []
    for pi, (label, _, _) in enumerate(editor.parts):
        # Sort points by time for readability
        pts = sorted(part_points.get(pi, []), key=lambda p: p["time_s"])

        aud_label_str = editor.parts_audacity_labels[pi] if pi < len(editor.parts_audacity_labels) else ""
        aud_bounds = editor.parts_audacity_bounds[pi] if pi < len(editor.parts_audacity_bounds) else None

        aud_out = None
        if aud_label_str or aud_bounds:
            aud_out = {
                "start": round(aud_bounds[0], 6) if aud_bounds else None,
                "end": round(aud_bounds[1], 6) if aud_bounds else None,
                "note": aud_label_str,
            }

        parts_out.append({
            "label": label,
            "notes": parts_notes.get(pi, ""),
            "points": pts,
            "audacity_label": aud_out,
        })

    data = {
        "filename": filename,
        "general_notes": general_notes,
        "had_audacity_labels": editor.had_audacity_labels,
        "audacity_label_txt": editor.raw_audacity_txt,
        "parts": parts_out,
        "unassigned_points": sorted(unassigned, key=lambda p: p["time_s"]),
    }

    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return path


# ── colors ────────────────────────────────────────────────────────────────────

PART_COLORS = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#000000",  # black
    "#17becf",  # cyan
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#ff7f0e",  # orange
    "#d62728",  # red
]


def _label_text_color(part_idx: int) -> str:
    h = PART_COLORS[part_idx % len(PART_COLORS)].lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 else "white"


# ── display options ───────────────────────────────────────────────────────────

_SHOW_HARMONICS = True       # show harmonic overlay lines
_SHOW_HARMONIC_DOTS = True  # show small dots on harmonic lines
_DOT_SIZE = 40               # control point scatter marker size
_FINE_DRAG_SCALE = 0.08      # Shift+drag attenuation (lower = finer)

# ── layout constants ──────────────────────────────────────────────────────────

_FIG_W = 12.0
_SPEC_H = 4.5
_ROW_H = 0.44
_ROW_GAP = 0.06
_BOTTOM_PAD = 0.14
_TOP_UI_PAD = 0.25  # gap between top button row and spectrogram bottom


def _compute_layout(n_parts: int):
    n_rows = n_parts + 3   # play + parts + add_part + notes/save
    ui_h = _BOTTOM_PAD + n_rows * (_ROW_H + _ROW_GAP) + _TOP_UI_PAD
    fig_h = _SPEC_H + ui_h
    bottom_frac = ui_h / fig_h
    row_h_frac = _ROW_H / fig_h

    def row_y(i: int) -> float:
        return (_BOTTOM_PAD + i * (_ROW_H + _ROW_GAP)) / fig_h

    return fig_h, bottom_frac, row_y, row_h_frac


# ── DraggableContour ──────────────────────────────────────────────────────────

class DraggableContour:
    """Interactive contour with draggable control points.

    part_indices == -1  →  unassigned (drawn cyan, no part membership).
    """

    def __init__(
        self,
        ax,
        time_s: np.ndarray,
        freq: np.ndarray,
        n_harmonics: int = 2,
        part_indices: np.ndarray | None = None,
        parts: list[tuple[str, float, float]] | None = None,
        parts_audacity_labels: list[str] | None = None,
        parts_audacity_bounds: list[tuple[float, float] | None] | None = None,
        had_audacity_labels: bool = False,
        raw_audacity_txt: str = "",
        orig_time_s: np.ndarray | None = None,
        orig_freq: np.ndarray | None = None,
    ):
        self.ax = ax
        self.time = np.array(time_s, dtype=float)
        self.freq = np.array(freq, dtype=float)
        self.n_harmonics = n_harmonics
        self.part_indices = np.asarray(part_indices) if part_indices is not None else None
        self.parts: list[tuple[str, float, float]] = list(parts) if parts else []
        self.parts_audacity_labels: list[str] = list(parts_audacity_labels) if parts_audacity_labels else [""] * len(self.parts)
        self.parts_audacity_bounds: list[tuple[float, float] | None] = list(parts_audacity_bounds) if parts_audacity_bounds else [None] * len(self.parts)
        self.had_audacity_labels: bool = had_audacity_labels
        self.raw_audacity_txt: str = raw_audacity_txt
        self.part_visible: dict[int, bool] = {}  # True = visible (default)
        self.orig_time_s = orig_time_s
        self.orig_freq = orig_freq
        self.dragging = None
        self.selected: int | None = None
        self.pick_radius = 0.06
        self._text_widgets: list = []
        self._keys_held: set[str] = set()
        self._drag_anchor_x = 0.0
        self._drag_anchor_y = 0.0
        self._drag_anchor_time = 0.0
        self._drag_anchor_freq = 0.0
        self._drag_last_x = 0.0
        self._drag_last_y = 0.0

        self._line_artists: list = []
        self._harmonics_lines: list = []
        self._harmonics_scatter = ax.scatter([], [], s=5, alpha=0.7, zorder=5, picker=False)
        self._draw_initial()
        self.scatter = ax.scatter(self.time, self.freq, s=_DOT_SIZE, zorder=10, edgecolors="black")
        self._update_scatter_colors()
        self._update_harmonics()

        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.ax.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.ax.figure.canvas.mpl_connect("key_release_event", self.on_key_release)

    def _part_color(self, part_idx: int) -> str:
        return PART_COLORS[part_idx % len(PART_COLORS)]

    def _assigned_part_indices(self) -> list[int]:
        if self.part_indices is None:
            return []
        return sorted(set(int(p) for p in self.part_indices if p >= 0))

    def _draw_initial(self):
        for ln in self._line_artists:
            ln.remove()
        self._line_artists.clear()

        if self.part_indices is not None and len(self.part_indices) == len(self.time):
            unassigned = self.part_indices == -1
            if np.any(unassigned):
                t, f = self.time[unassigned], self.freq[unassigned]
                order = np.argsort(t)
                ln, = self.ax.plot(t[order], f[order], "c-", linewidth=2, zorder=6)
                self._line_artists.append(ln)
            for pi in self._assigned_part_indices():
                mask = self.part_indices == pi
                t, f = self.time[mask], self.freq[mask]
                order = np.argsort(t)
                ln, = self.ax.plot(t[order], f[order], "-", color=self._part_color(pi), linewidth=2, zorder=6)
                ln.set_visible(self.part_visible.get(pi, True))
                self._line_artists.append(ln)
        else:
            ln, = self.ax.plot(self.time, self.freq, "c-", linewidth=2, zorder=6)
            self._line_artists.append(ln)

    def _get_nearest(self, x, y) -> int | None:
        if x is None or y is None:
            return None
        dx = (self.time - x) / (self.ax.get_xlim()[1] - self.ax.get_xlim()[0] + 1e-9)
        dy = (self.freq - y) / (self.ax.get_ylim()[1] - self.ax.get_ylim()[0] + 1e-9)
        dist = np.sqrt(dx**2 + dy**2)
        i = int(np.argmin(dist))
        return i if dist[i] < self.pick_radius else None

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        i = self._get_nearest(event.xdata, event.ydata)
        self.dragging = i
        self.selected = i
        if i is not None:
            self._drag_anchor_x = event.xdata
            self._drag_anchor_y = event.ydata
            self._drag_anchor_time = self.time[i]
            self._drag_anchor_freq = self.freq[i]
            self._drag_last_x = event.xdata
            self._drag_last_y = event.ydata
        self._update_scatter_colors()

    def on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax:
            return
        y_min, y_max = self.ax.get_ylim()
        ctrl = "control" in self._keys_held
        shift = "shift" in self._keys_held

        if ctrl and self.part_indices is not None:
            # Move all points in the same part by delta
            dx = event.xdata - self._drag_last_x
            dy = event.ydata - self._drag_last_y
            pi = int(self.part_indices[self.dragging])
            mask = self.part_indices == pi if pi >= 0 else np.array([self.dragging])
            self.time[mask] += dx
            self.freq[mask] = np.clip(self.freq[mask] + dy, y_min, y_max)
        elif shift:
            # Fine control: attenuate movement relative to drag start
            total_dx = event.xdata - self._drag_anchor_x
            total_dy = event.ydata - self._drag_anchor_y
            self.time[self.dragging] = self._drag_anchor_time + total_dx * _FINE_DRAG_SCALE
            self.freq[self.dragging] = np.clip(
                self._drag_anchor_freq + total_dy * _FINE_DRAG_SCALE, y_min, y_max
            )
        else:
            self.time[self.dragging] = event.xdata
            self.freq[self.dragging] = np.clip(event.ydata, y_min, y_max)

        self._drag_last_x = event.xdata
        self._drag_last_y = event.ydata
        self._redraw()

    def on_release(self, event):
        self.dragging = None

    def on_key_release(self, event):
        self._keys_held.discard(event.key)

    def on_key(self, event):
        self._keys_held.add(event.key)
        for tw in self._text_widgets:
            if tw.capturekeystrokes:
                return
        if event.key == "`":
            self._expand_from_orig()
            return
        if self.selected is None:
            return
        if event.key in ("d", "D"):
            if len(self.time) > 1:
                self.time = np.delete(self.time, self.selected)
                self.freq = np.delete(self.freq, self.selected)
                if self.part_indices is not None:
                    self.part_indices = np.delete(self.part_indices, self.selected)
                self.selected = min(self.selected, len(self.time) - 1) if self.time.size else None
                self._full_redraw()
        elif event.key in ("a", "A"):
            t_sel, f_sel = self.time[self.selected], self.freq[self.selected]
            if self.selected < len(self.time) - 1:
                t_new = (t_sel + self.time[self.selected + 1]) / 2
                f_new = (f_sel + self.freq[self.selected + 1]) / 2
            else:
                t_new = t_sel + 0.01
                f_new = f_sel
            insert_at = self.selected + 1
            self.time = np.insert(self.time, insert_at, t_new)
            self.freq = np.insert(self.freq, insert_at, f_new)
            if self.part_indices is not None:
                self.part_indices = np.insert(self.part_indices, insert_at, self.part_indices[self.selected])
            self.selected = insert_at
            self._full_redraw()

    def _expand_from_orig(self):
        """`  key: replace a 2-dot part with the original f0 contour."""
        if self.orig_time_s is None or self.orig_freq is None:
            return
        if self.part_indices is not None and self.selected is not None:
            pi = int(self.part_indices[self.selected])
            if pi < 0:
                return
            dot_indices = np.where(self.part_indices == pi)[0]
            if len(dot_indices) != 2:
                return
        elif self.part_indices is None and len(self.time) == 2:
            pi = None
            dot_indices = np.array([0, 1])
        else:
            return

        t_start = float(np.min(self.time[dot_indices]))
        t_end = float(np.max(self.time[dot_indices]))

        orig_mask = (self.orig_time_s >= t_start) & (self.orig_time_s <= t_end)
        t_orig = self.orig_time_s[orig_mask]
        f_orig = self.orig_freq[orig_mask]
        if len(t_orig) < 2:
            return

        n_new = max(2, int((t_end - t_start) / _MIN_SPACING_S) + 1)
        t_new, f_new = downsample_control_points(t_orig, f_orig, n_new)

        keep = np.ones(len(self.time), dtype=bool)
        keep[dot_indices] = False
        self.time = np.concatenate([self.time[keep], t_new])
        self.freq = np.concatenate([self.freq[keep], f_new])
        if self.part_indices is not None:
            self.part_indices = np.concatenate([
                self.part_indices[keep],
                np.full(len(t_new), pi, dtype=int),
            ])
        self.selected = None
        self._full_redraw()

    def _update_scatter_colors(self):
        if self.part_indices is not None and len(self.part_indices) == len(self.time):
            face_colors, edge_colors = [], []
            for i, pi in enumerate(self.part_indices):
                part_vis = self.part_visible.get(int(pi), True) if pi >= 0 else True
                alpha = 1.0 if part_vis else 0.0
                if i == self.selected:
                    fc = (*to_rgba("orange")[:3], alpha)
                elif pi == -1:
                    fc = (*to_rgba("cyan")[:3], alpha)
                else:
                    fc = (*to_rgba(self._part_color(int(pi)))[:3], alpha)
                face_colors.append(fc)
                edge_colors.append((0.0, 0.0, 0.0, alpha))
            self.scatter.set_facecolors(face_colors)
            self.scatter.set_edgecolors(edge_colors)
        else:
            self.scatter.set_facecolors(["orange" if i == self.selected else "yellow" for i in range(len(self.time))])

    def _update_harmonics(self):
        for ln in self._harmonics_lines:
            ln.remove()
        self._harmonics_lines.clear()

        if not _SHOW_HARMONICS or self.n_harmonics <= 0 or len(self.time) == 0:
            self._harmonics_scatter.set_offsets(np.empty((0, 2)))
            self._harmonics_scatter.set_facecolors([])
            return

        h_times, h_freqs, h_colors = [], [], []
        use_parts = self.part_indices is not None and len(self.part_indices) == len(self.time)

        for k in range(2, self.n_harmonics + 2):
            if use_parts:
                for pi in self._assigned_part_indices():
                    if not self.part_visible.get(pi, True):
                        continue
                    idxs = [i for i in np.where(self.part_indices == pi)[0] if not np.isnan(self.freq[i]) and self.freq[i] > 0]
                    if not idxs:
                        continue
                    t_k = self.time[idxs]
                    f_k = k * self.freq[idxs]
                    order = np.argsort(t_k)
                    t_k, f_k = t_k[order], f_k[order]
                    color = self._part_color(pi)
                    ln, = self.ax.plot(t_k, f_k, "-", color=color, linewidth=1, alpha=0.6, zorder=4)
                    self._harmonics_lines.append(ln)
                    h_times.extend(t_k); h_freqs.extend(f_k); h_colors.extend([color] * len(t_k))
            else:
                valid = [(t, f) for t, f in zip(self.time, self.freq) if not np.isnan(f) and f > 0]
                if not valid:
                    continue
                t_k = np.array([v[0] for v in valid])
                f_k = np.array([k * v[1] for v in valid])
                order = np.argsort(t_k)
                t_k, f_k = t_k[order], f_k[order]
                ln, = self.ax.plot(t_k, f_k, "y-", linewidth=1, alpha=0.6, zorder=4)
                self._harmonics_lines.append(ln)
                h_times.extend(t_k); h_freqs.extend(f_k); h_colors.extend(["yellow"] * len(t_k))

        if h_times and _SHOW_HARMONIC_DOTS:
            self._harmonics_scatter.set_offsets(np.column_stack([h_times, h_freqs]))
            self._harmonics_scatter.set_facecolors(h_colors)
            self._harmonics_scatter.set_edgecolors("black")
        else:
            self._harmonics_scatter.set_offsets(np.empty((0, 2)))

    def toggle_part_visible(self, pi: int):
        self.part_visible[pi] = not self.part_visible.get(pi, True)
        self._redraw()

    def _offsets(self):
        if len(self.time) == 0:
            return np.empty((0, 2))
        return np.column_stack([self.time, self.freq])

    def _redraw(self):
        for ln in self._line_artists:
            ln.remove()
        self._line_artists.clear()
        self._draw_initial()
        self.scatter.set_offsets(self._offsets())
        self._update_scatter_colors()
        self._update_harmonics()
        self.ax.figure.canvas.draw_idle()

    def _full_redraw(self):
        for ln in self._line_artists:
            ln.remove()
        self._line_artists.clear()
        self._draw_initial()
        self.scatter.set_offsets(self._offsets())
        self.scatter.set_sizes([_DOT_SIZE] * len(self.time))
        self._update_scatter_colors()
        self._update_harmonics()
        self.ax.figure.canvas.draw_idle()

    def get_contour(self) -> tuple[np.ndarray, np.ndarray]:
        return self.time.copy(), self.freq.copy()


# ── ContourEditorUI ───────────────────────────────────────────────────────────

class ContourEditorUI:
    """Dynamic control panel below the spectrogram.

    Rows (bottom → top):
      0  General notes + Save button
      1  + Add Part button
      2…n+1  One row per part: [label] [notes textbox] [✕]
      n+2  ▶ Play / ■ Stop button
    """

    def __init__(
        self,
        fig,
        ax_spec,
        editor: DraggableContour,
        y_audio: np.ndarray,
        sr_audio: int,
        save_folder: str,
        filename: str,
        init_parts_notes: dict[int, str] | None = None,
        init_general_notes: str = "",
        on_next=None,  # optional callable() — when set, a "Next →" button is shown
    ):
        self.fig = fig
        self.ax_spec = ax_spec
        self.editor = editor
        self.y_audio = y_audio
        self.sr_audio = sr_audio
        self.save_folder = save_folder
        self.filename = filename
        self._on_next_cb = on_next

        self._playing = [False]
        self._btn_play: Button | None = None
        self._txt_general: TextBox | None = None
        self._txt_parts: dict[int, TextBox] = {}

        self._general_notes: str = init_general_notes
        self._parts_notes: dict[int, str] = dict(init_parts_notes) if init_parts_notes else {}

        self._widget_axes: list = []
        self._widgets: list = []
        self._spec_labels: list = []
        self._part_label_btns: dict[int, Button] = {}

        self._build()

    def _flush_notes(self):
        if self._txt_general is not None:
            self._general_notes = self._txt_general.text
        for pi, tw in self._txt_parts.items():
            self._parts_notes[pi] = tw.text

    def _build(self):
        self._flush_notes()

        for wax in self._widget_axes:
            self.fig.delaxes(wax)
        self._widget_axes.clear()
        self._widgets.clear()
        self._txt_parts = {}
        self._txt_general = None
        self._part_label_btns = {}

        for t in self._spec_labels:
            t.remove()
        self._spec_labels.clear()

        n_parts = len(self.editor.parts)
        fig_h, bottom_frac, row_y, row_h_frac = _compute_layout(n_parts)

        self.fig.set_size_inches(_FIG_W, fig_h, forward=True)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=bottom_frac)

        # ── Row 0: General notes + Save [+ Next] ─────────────────────────
        y = row_y(0)
        if self._on_next_cb is not None:
            ax_save = self.fig.add_axes([0.76, y, 0.08, row_h_frac])
            ax_next = self.fig.add_axes([0.85, y, 0.10, row_h_frac])
            btn_next = Button(ax_next, "Next →", color="#226622", hovercolor="#338833")
            btn_next.label.set_color("white")
            btn_next.on_clicked(self._on_next)
            self._widget_axes.append(ax_next)
            self._widgets.append(btn_next)
            notes_w = 0.60
        else:
            ax_save = self.fig.add_axes([0.83, y, 0.10, row_h_frac])
            notes_w = 0.67
        btn_save = Button(ax_save, "Save")
        btn_save.on_clicked(self._on_save)

        ax_gn = self.fig.add_axes([0.13, y, notes_w, row_h_frac])
        self._txt_general = TextBox(ax_gn, "Notes:", initial=self._general_notes)
        self._widget_axes += [ax_save, ax_gn]
        self._widgets += [btn_save, self._txt_general]

        # ── Row 1: Add Part ───────────────────────────────────────────────
        y = row_y(1)
        ax_add = self.fig.add_axes([0.05, y, 0.18, row_h_frac])
        btn_add = Button(ax_add, "+ Add Part")
        btn_add.on_clicked(self._on_add_part)
        self._widget_axes.append(ax_add)
        self._widgets.append(btn_add)

        # ── Rows 2…n+1: Part rows (P1 closest to spectrogram = highest) ──
        for row_i, pi in enumerate(reversed(range(n_parts))):
            y = row_y(2 + row_i)
            label, _, _ = self.editor.parts[pi]
            color = PART_COLORS[pi % len(PART_COLORS)]
            txt_color = _label_text_color(pi)

            visible = self.editor.part_visible.get(pi, True)
            btn_color = color if visible else "#555555"
            ax_lbl = self.fig.add_axes([0.05, y, 0.07, row_h_frac])
            btn_lbl = Button(ax_lbl, label, color=btn_color, hovercolor=btn_color)
            btn_lbl.label.set_fontweight("bold")
            btn_lbl.label.set_color(txt_color if visible else "#aaaaaa")
            self._part_label_btns[pi] = btn_lbl

            def _make_toggle(idx):
                def cb(e):
                    self.editor.toggle_part_visible(idx)
                    vis = self.editor.part_visible.get(idx, True)
                    base = PART_COLORS[idx % len(PART_COLORS)]
                    c = base if vis else "#555555"
                    tc = _label_text_color(idx) if vis else "#aaaaaa"
                    self._part_label_btns[idx].ax.set_facecolor(c)
                    self._part_label_btns[idx].label.set_color(tc)
                    self.fig.canvas.draw_idle()
                return cb
            btn_lbl.on_clicked(_make_toggle(pi))

            ax_pn = self.fig.add_axes([0.14, y, 0.66, row_h_frac])
            txt_pn = TextBox(ax_pn, "", initial=self._parts_notes.get(pi, ""))
            self._txt_parts[pi] = txt_pn

            ax_rm = self.fig.add_axes([0.83, y, 0.06, row_h_frac])
            btn_rm = Button(ax_rm, "✕", color="#aa2222", hovercolor="#cc3333")
            btn_rm.label.set_color("white")

            def _make_rm(idx):
                def cb(e): self._on_remove_part(idx)
                return cb
            btn_rm.on_clicked(_make_rm(pi))

            self._widget_axes += [ax_lbl, ax_pn, ax_rm]
            self._widgets += [btn_lbl, txt_pn, btn_rm]

            txt = self.ax_spec.text(
                0.01, 0.99 - 0.06 * (n_parts - 1 - pi),
                label, fontsize=10, color=color, fontweight="bold",
                va="top", ha="left", transform=self.ax_spec.transAxes,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, edgecolor="none"),
            )
            self._spec_labels.append(txt)

        # ── Top row: Play + Zoom ───────────────────────────────────────────
        y = row_y(n_parts + 2)
        ax_play = self.fig.add_axes([0.05, y, 0.12, row_h_frac])
        btn_play = Button(ax_play, "▶ Play")
        btn_play.on_clicked(self._on_play)
        self._btn_play = btn_play
        self._widget_axes.append(ax_play)
        self._widgets.append(btn_play)

        ax_zin = self.fig.add_axes([0.19, y, 0.06, row_h_frac])
        btn_zin = Button(ax_zin, "+ Zoom")
        btn_zin.on_clicked(self._on_zoom_in)
        self._widget_axes.append(ax_zin)
        self._widgets.append(btn_zin)

        ax_zout = self.fig.add_axes([0.26, y, 0.06, row_h_frac])
        btn_zout = Button(ax_zout, "- Zoom")
        btn_zout.on_clicked(self._on_zoom_out)
        self._widget_axes.append(ax_zout)
        self._widgets.append(btn_zout)

        self.editor._text_widgets = [self._txt_general] + list(self._txt_parts.values())
        self.fig.canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────

    def _on_zoom_in(self, event):
        _, y_max = self.ax_spec.get_ylim()
        self.ax_spec.set_ylim(0, max(y_max / 2, 500))
        self.fig.canvas.draw_idle()

    def _on_zoom_out(self, event):
        _, y_max = self.ax_spec.get_ylim()
        self.ax_spec.set_ylim(0, min(y_max * 2, 24000))
        self.fig.canvas.draw_idle()

    def _on_play(self, event):
        if not _HAS_SOUNDDEVICE:
            print("sounddevice not installed. Run: pip install sounddevice")
            return
        if self._playing[0]:
            sd.stop()
            self._playing[0] = False
            self._btn_play.label.set_text("▶ Play")
            self.fig.canvas.draw_idle()
        else:
            self._playing[0] = True
            self._btn_play.label.set_text("■ Stop")
            self.fig.canvas.draw_idle()

            def _play():
                sd.play(self.y_audio, self.sr_audio)
                sd.wait()
                self._playing[0] = False
                self._btn_play.label.set_text("▶ Play")
                self.fig.canvas.draw_idle()

            threading.Thread(target=_play, daemon=True).start()

    def _on_next(self, event):
        self._flush_notes()
        path = save_annotations(
            self.save_folder, self.filename,
            self.editor, self._general_notes, self._parts_notes,
        )
        print(f"Saved → {path}")
        plt.close(self.fig)
        if self._on_next_cb is not None:
            self._on_next_cb()

    def _on_save(self, event):
        self._flush_notes()
        path = save_annotations(
            self.save_folder, self.filename,
            self.editor, self._general_notes, self._parts_notes,
        )
        print(f"Saved → {path}")
        if self.editor.parts:
            for pi, (label, _, _) in enumerate(self.editor.parts):
                count = int(np.sum(self.editor.part_indices == pi)) if self.editor.part_indices is not None else 0
                print(f"  {label}: {count} dots  notes: {self._parts_notes.get(pi, '')!r}")
        else:
            print(f"  Total dots: {len(self.editor.time)}")

    def _on_add_part(self, event):
        xlim = self.ax_spec.get_xlim()
        t_center = (xlim[0] + xlim[1]) / 2
        t1, t2 = t_center - 0.05, t_center + 0.05
        ylim = self.ax_spec.get_ylim()
        f_mid = ylim[0] + (ylim[1] - ylim[0]) * 0.25

        new_pi = len(self.editor.parts)

        if self.editor.part_indices is None:
            self.editor.part_indices = np.full(len(self.editor.time), -1, dtype=int)

        self.editor.time = np.append(self.editor.time, [t1, t2])
        self.editor.freq = np.append(self.editor.freq, [f_mid, f_mid])
        self.editor.part_indices = np.append(self.editor.part_indices, [new_pi, new_pi])
        self.editor.parts.append((f"P{new_pi + 1}", t1, t2))
        self.editor.parts_audacity_labels.append("")
        self.editor.parts_audacity_bounds.append(None)

        self.editor._full_redraw()
        self._build()

    def _on_remove_part(self, part_idx: int):
        self._flush_notes()

        if self.editor.part_indices is not None:
            keep = self.editor.part_indices != part_idx
            self.editor.time = self.editor.time[keep]
            self.editor.freq = self.editor.freq[keep]
            new_pi = self.editor.part_indices[keep].copy()
            new_pi[new_pi > part_idx] -= 1
            self.editor.part_indices = new_pi

        # Shift notes and visibility dicts
        self._parts_notes = {
            (pi if pi < part_idx else pi - 1): notes
            for pi, notes in self._parts_notes.items()
            if pi != part_idx
        }
        self.editor.part_visible = {
            (pi if pi < part_idx else pi - 1): vis
            for pi, vis in self.editor.part_visible.items()
            if pi != part_idx
        }

        self.editor.parts.pop(part_idx)
        self.editor.parts_audacity_labels.pop(part_idx)
        self.editor.parts_audacity_bounds.pop(part_idx)

        if self.editor.selected is not None and self.editor.selected >= len(self.editor.time):
            self.editor.selected = None

        self.editor._full_redraw()
        self._build()


# ── run_editor ────────────────────────────────────────────────────────────────

def run_editor(
    folder_path: str,
    filename: str | None = None,
    n_control_points: int = 25,
    n_harmonics: int = 2,
    save_folder: str | None = None,
    spects_folder: str = "data/ford_paper_spects",
    n_fft: int = 1024,
    hop_length: int = 128,
    on_next=None,  # optional callable() invoked when "Next →" is clicked
):
    y, sr, S, time_s, freq, name = load_audio_and_pitch(folder_path, filename, n_fft=n_fft, hop_length=hop_length)

    # Determine filename (may have been auto-detected)
    if filename is None:
        filename = os.path.splitext(name)[0]

    # Save folder defaults to audio folder
    if save_folder is None:
        save_folder = os.path.abspath(folder_path)

    # ── Load annotations: priority: saved JSON > audacity .txt > empty ───
    parts: list[tuple[str, float, float]] = []
    parts_audacity_labels: list[str] = []
    parts_audacity_bounds: list[tuple[float, float] | None] = []
    had_audacity_labels = False
    raw_audacity_txt = ""
    part_indices: np.ndarray | None = None
    init_parts_notes: dict[int, str] = {}
    init_general_notes = ""

    saved = load_saved_annotations(save_folder, filename)
    if saved is not None:
        print(f"Loaded saved annotations: {save_folder}/{filename}_contour.json")
        ctrl_time = saved["ctrl_time"]
        ctrl_freq = saved["ctrl_freq"]
        part_indices = saved["part_indices"]
        parts = saved["parts"]
        parts_audacity_labels = saved["parts_audacity_labels"]
        parts_audacity_bounds = saved["parts_audacity_bounds"]
        had_audacity_labels = saved["had_audacity_labels"]
        raw_audacity_txt = saved["raw_audacity_txt"]
        init_parts_notes = saved["parts_notes"]
        init_general_notes = saved["general_notes"]
    else:
        aud = parse_old_audacity_parts(folder_path, filename)
        if aud is not None:
            parts, parts_audacity_labels = aud
            # audacity bounds = the start/end from the .txt annotation
            parts_audacity_bounds = [(s, e) for _, s, e in parts]
            had_audacity_labels = True
            txt_path = _find_parts_txt(folder_path, filename)
            if txt_path:
                with open(txt_path) as fh:
                    raw_audacity_txt = fh.read()
            print(f"Loaded {len(parts)} part(s) from Audacity labels")
            ctrl_time, ctrl_freq, part_indices_list = get_control_points_by_parts(time_s, freq, parts, n_control_points)
            part_indices = np.array(part_indices_list)
        else:
            print("No existing annotations found — starting empty.")
            ctrl_time, ctrl_freq = np.array([]), np.array([])

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(_FIG_W, _SPEC_H + 2.0))
    gs = GridSpec(1, 5, figure=fig, wspace=0.08)
    ax = fig.add_subplot(gs[0, :4])     # main spectrogram (4/5 width)
    ax_ref = fig.add_subplot(gs[0, 4])  # reference image  (1/5 width)

    img = librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        sr=sr, hop_length=hop_length, n_fft=n_fft, x_axis="time", y_axis="linear", ax=ax,
    )
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(name, fontsize=9, color="white", pad=4)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_ylim(0, sr / 2)
    ax.autoscale(enable=False, axis="y")

    # ── Reference image panel ─────────────────────────────────────────────
    call_type = filename.split("-")[0]
    ref_path = os.path.join(spects_folder, f"{call_type}_paper_spect.png")
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    ax_ref.set_title(call_type, fontsize=8, pad=3, color="white")
    ax_ref.set_facecolor("#111111")
    if os.path.exists(ref_path):
        ax_ref.imshow(mpimg.imread(ref_path))
    else:
        ax_ref.text(0.5, 0.5, f"Paper spect\nnot found\nfor {call_type}",
                    ha="center", va="center", fontsize=7, color="gray",
                    transform=ax_ref.transAxes)

    editor = DraggableContour(
        ax, ctrl_time, ctrl_freq,
        n_harmonics=n_harmonics,
        part_indices=part_indices,
        parts=parts,
        parts_audacity_labels=parts_audacity_labels,
        parts_audacity_bounds=parts_audacity_bounds,
        had_audacity_labels=had_audacity_labels,
        raw_audacity_txt=raw_audacity_txt,
        orig_time_s=time_s,
        orig_freq=freq,
    )
    _ = ContourEditorUI(
        fig, ax, editor, y, sr,
        save_folder=save_folder,
        filename=filename,
        init_parts_notes=init_parts_notes,
        init_general_notes=init_general_notes,
        on_next=on_next,
    )

    plt.show()
    return editor.get_contour()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive contour editor")
    parser.add_argument("--folder", type=str, default="data/ford-catalogue/Northern Resident/media/A_clan")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--points", type=int, default=25, help="Max control points per part (min spacing 0.1 s)")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--save-folder", type=str, default=None, help="Folder for saved JSONs (default: same as audio)")
    parser.add_argument("--spects-folder", type=str, default="data/ford_paper_spects", help="Folder with *_paper_spect.png reference images")
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=128)
    args = parser.parse_args()
    run_editor(args.folder, args.file, args.points, args.harmonics, args.save_folder, args.spects_folder, args.nfft, args.hop)
