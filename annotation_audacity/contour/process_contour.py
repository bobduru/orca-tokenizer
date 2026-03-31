from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import librosa.display
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).parent
ANNOTATIONS_DIR = _HERE / "../../annotations"
CATALOGUE_ROOT  = _HERE / "../../"

# ── catalogue & audio index ───────────────────────────────────────────────────
catalogue_df = pd.read_csv(_HERE / "../../data/ford-catalogue/online_catalogue.csv")

_audio_index = {
    Path(fp).stem: CATALOGUE_ROOT / fp
    for fp in catalogue_df["audio_fp"]
}


# ── part type detection ───────────────────────────────────────────────────────
_PART_TYPE_KEYWORDS = ["TOO TIGHT", "BLUR", "VERTICAL", "GAP"]

def _detect_part_type(notes: str) -> str:
    """Return 'TOO TIGHT' | 'BLUR' | 'VERTICAL' | 'GAP' | 'PRECISE'."""
    for kw in _PART_TYPE_KEYWORDS:
        if kw in notes.upper():
            return kw
    return "PRECISE"


# ── audio lookup ──────────────────────────────────────────────────────────────
def find_audio(filename: str) -> Path | None:
    """Look up the audio path for a contour filename via the catalogue index."""
    stem = Path(filename).stem.removesuffix("_contour")
    return _audio_index.get(stem)


# ── loading ───────────────────────────────────────────────────────────────────
def load_contour(filename: str) -> dict | None:
    """
    Load a contour JSON by base filename (e.g. 'N01i-A1-2') or full path.
    Returns the parsed dict with a 'parts_parsed' key added:
        parts_parsed: list of dicts with 'label', 'notes', 'type', 'times', 'freqs'

    'type' is one of: 'PRECISE' | 'TOO TIGHT' | 'BLUR' | 'VERTICAL' | 'GAP'
    For non-PRECISE parts, points are time markers only; frequency is unreliable.

    Returns None if the contour is marked for skipping.
    """
    path = Path(filename)
    if not path.exists():
        stem = Path(filename).stem.removesuffix("_contour")
        path = ANNOTATIONS_DIR / f"{stem}_contour.json"

    with open(path) as fh:
        data = json.load(fh)

    if "skip" in data.get("general_notes", "").lower():
        print(f"Skipping {filename}: {data['general_notes']}")
        return None

    parts_parsed = []
    for part in data.get("parts", []):
        pts = part.get("points", [])
        if not pts:
            print(f"Skipping empty part {part['label']} in {filename}")
            continue
        times = np.array([p["time_s"] for p in pts])
        freqs = np.array([p["freq_hz"] for p in pts])
        notes = part.get("notes", "")
        parts_parsed.append({
            "label":        part["label"],
            "notes":        notes,
            "type":         _detect_part_type(notes),
            "audacity_note": part.get("audacity_label", {}).get("note", "") if part.get("audacity_label") else "",
            "times":        times,
            "freqs":        freqs,
        })

    data["parts_parsed"] = parts_parsed
    data["segments"]     = classify_segments(data)
    data["segments_2nd"] = parse_second_order(data["segments"], flat_tolerance_seconds=0.1)
    data["segments_3rd"] = parse_third_order(data["segments_2nd"])
    return data


def load_all_contours() -> list[dict]:
    """Load every *_contour.json from ANNOTATIONS_DIR."""
    contours = []
    for path in sorted(ANNOTATIONS_DIR.glob("*_contour.json")):
        data = load_contour(str(path))
        if data is not None:
            contours.append(data)
    print(f"Loaded {len(contours)} contours")
    return contours


cycle = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

# ── plotting ──────────────────────────────────────────────────────────────────
_PART_COLORS = [
    cycle[0],
    cycle[1],
    cycle[2],
    cycle[3],
    cycle[4],
    cycle[5],
    cycle[6],
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#17becf",  # teal
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
]


def plot_contour(
    contour: dict,
    *,
    with_spectrogram: bool = True,
    segments: list[dict] | None = None,
    n_harmonics: int = 1,
    n_fft: int = 1024,
    hop_length: int = 128,
    figsize: tuple = (14, 5),
    cmap_spectrogram: str = "magma",
) -> None:
    """
    Plot the annotated contour.

    Parameters
    ----------
    contour          : output of load_contour()
    with_spectrogram : draw spectrogram background (loads audio); falls back to
                       white background if audio is not found
    segments         : output of classify_segments() or parse_second_order() —
                       if provided, draw a colour-coded segment bar below the contour
    n_harmonics      : if > 1, also plot integer multiples of each contour frequency
    """
    parts    = contour["parts_parsed"]
    filename = contour["filename"]

    # ── figure layout ─────────────────────────────────────────────────────────
    if segments is not None:
        fig, (ax, ax_seg) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [5, 1]},
            sharex=False,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_seg = None

    # ── axis bounds ───────────────────────────────────────────────────────────
    t_start = min(p["times"][0]  for p in parts)
    t_end   = max(p["times"][-1] for p in parts)
    f_min   = min(p["freqs"].min() for p in parts)
    f_max   = max(p["freqs"].max() for p in parts)
    if n_harmonics > 1:
        f_max = f_max * n_harmonics
    pad_t = (t_end - t_start) * 0.05
    pad_f = (f_max - f_min) * 0.15

    # ── background ────────────────────────────────────────────────────────────
    if with_spectrogram:
        audio_path = find_audio(filename)
        if audio_path is None:
            print(f"[warn] audio not found for {filename}, using white background")
            with_spectrogram = False
        else:
            y, sr = librosa.load(str(audio_path))
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            librosa.display.specshow(
                librosa.amplitude_to_db(S, ref=np.max),
                sr=sr, n_fft=n_fft, hop_length=hop_length,
                x_axis="time", y_axis="linear",
                ax=ax, cmap=cmap_spectrogram,
            )
            ax.set_ylim(0, sr / 2)
            ax.set_xlim(t_start - pad_t, t_end + pad_t)

    if not with_spectrogram:
        ax.set_facecolor("white")
        ax.set_xlim(t_start - pad_t, t_end + pad_t)
        ax.set_ylim(f_min - pad_f, f_max + pad_f)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    # ── contour lines & dots ──────────────────────────────────────────────────
    for i, part in enumerate(parts):
        color = _PART_COLORS[i % len(_PART_COLORS)]
        for h in range(1, n_harmonics + 1):
            freqs_h = part["freqs"] * h
            alpha   = 1.0 if h == 1 else max(0.3, 1.0 - 0.4 * (h - 1))
            lw      = 2   if h == 1 else 1
            ax.plot(part["times"], freqs_h, color=color, linewidth=lw, alpha=alpha, zorder=3, marker="o", markersize=4)
            if h == 1:
                # ax.scatter(part["times"], freqs_h, color=color, s=40, zorder=4)
                ax.annotate(
                    part["label"],
                    xy=(part["times"][0], freqs_h[0]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold",
                )

    ax.set_title(filename)

    # ── segment bar ───────────────────────────────────────────────────────────
    if segments is not None and ax_seg is not None:
        _draw_segment_bar(ax_seg, segments, t_start - pad_t, t_end + pad_t)
        fig.subplots_adjust(hspace=0.05)

    plt.tight_layout()
    plt.show()


def _draw_segment_bar(
    ax,
    segments: list[dict],
    xlim_left: float,
    xlim_right: float,
) -> None:
    """Draw a colour-coded rectangle bar with segment labels."""
    _SEG_COLORS = {
        "UP":     "#4caf50",
        "DOWN":   "#f44336",
        "FLAT":   "#90a4ae",
        "PEAK":   "#1976d2",
        "VALLEY": "#00bcd4",
    }

    for seg in segments:
        color = _SEG_COLORS.get(seg["label"], "#cccccc")
        ax.barh(
            y=0, left=seg["t_start"], width=seg["t_end"] - seg["t_start"],
            height=1, color=color, edgecolor="white", linewidth=0.5, align="edge",
        )
        mid = (seg["t_start"] + seg["t_end"]) / 2
        ax.text(
            mid, 0.5, seg["label"],
            ha="center", va="center", fontsize=7, color="white", fontweight="bold",
        )

    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # legend_patches = [
    #     mpatches.Patch(color=c, label=l)
    #     for l, c in _SEG_COLORS.items()
    # ]
    # ax.legend(handles=legend_patches, loc="center right", fontsize=7, framealpha=0.8)


# ── segment classification ────────────────────────────────────────────────────
def _runs_to_segments(
    times,
    freqs,
    *,
    abs_threshold_hz: float,
    pct_threshold: float,
    noise_floor_hz: float,
    max_merge_duration: float,
) -> list[dict]:
    """
    Build runs of consecutive same-direction diffs, then classify each run
    as UP / DOWN / FLAT.

    A run ends when:
      - the direction (sign) changes, OR
      - the individual segment duration >= max_merge_duration
        (long segments are always their own run, never absorbed into a neighbour)

    A run is classified UP/DOWN only if its total |Δf| >=
    max(abs_threshold_hz, pct_threshold × part_range). Otherwise: FLAT.

    Finally, adjacent runs with the same label are merged.
    """
    part_range = freqs.max() - freqs.min()
    threshold  = max(abs_threshold_hz, pct_threshold * part_range)

    diffs = np.diff(freqs)
    signs = np.where(diffs > noise_floor_hz, 1, np.where(diffs < -noise_floor_hz, -1, 0))

    # Build runs ──────────────────────────────────────────────────────────────
    runs = []  # [start_idx, end_idx, sign]
    i = 0
    while i < len(signs):
        j = i
        while j < len(signs):
            if signs[j] != signs[i]:
                break
            if times[j + 1] - times[j] >= max_merge_duration:
                break
            j += 1
        if j == i:
            j += 1
        runs.append([i, j, signs[i]])
        i = j

    # Label each run ──────────────────────────────────────────────────────────
    labeled = []
    for start, end, sign in runs:
        total_df = freqs[end] - freqs[start]
        if sign == 0 or abs(total_df) < threshold:
            label = "FLAT"
        elif total_df > 0:
            label = "UP"
        else:
            label = "DOWN"
        labeled.append([start, end, label])

    # Merge adjacent same-label runs ──────────────────────────────────────────
    merged = [labeled[0][:]]
    for start, end, label in labeled[1:]:
        if label == merged[-1][2]:
            merged[-1][1] = end
        else:
            merged.append([start, end, label])

    # Convert to dicts ────────────────────────────────────────────────────────
    segments = []
    for start, end, label in merged:
        t0, t1   = times[start], times[end]
        f0, f1   = freqs[start], freqs[end]
        delta_f  = f1 - f0
        delta_t  = t1 - t0
        segments.append({
            "t_start":     t0,
            "t_end":       t1,
            "f_start":     f0,
            "f_end":       f1,
            "delta_f":     delta_f,
            "delta_t":     delta_t,
            "slope":       delta_f / delta_t if delta_t > 0 else 0.0,
            "abs_delta_f": abs(delta_f),
            "label":       label,
            "points":      list(zip(times[start : end + 1], freqs[start : end + 1])),
        })
    return segments


def classify_segments(
    contour: dict,
    *,
    abs_threshold_hz: float = 50.0,
    pct_threshold: float = 0.10,
    noise_floor_hz: float = 5.0,
    max_merge_duration: float = 0.1,
) -> list[dict]:
    """
    Classify contour parts into segments.

    For 'PRECISE' parts: see _runs_to_segments.
    For non-PRECISE parts ('too tight', 'blur', 'vertical', 'gap'): emit one
    segment spanning the whole part with label = part type. Frequency is ignored.

    Parameters
    ----------
    abs_threshold_hz   : minimum absolute Δf (Hz) for a run to count as UP/DOWN.
    pct_threshold      : minimum Δf as fraction of the part's freq range.
    noise_floor_hz     : diffs smaller than this are treated as zero (rounding noise).
    max_merge_duration : segments longer than this (seconds) always end the current
                         run and are never absorbed into a neighbour.
    """
    segments = []
    for part in contour["parts_parsed"]:
        times     = part["times"]
        freqs     = part["freqs"]
        part_type = part["type"]

        if len(times) < 2:
            continue

        if part_type != "PRECISE":
            segments.append({
                "t_start":     times[0],
                "t_end":       times[-1],
                "f_start":     None,
                "f_end":       None,
                "delta_f":     None,
                "delta_t":     times[-1] - times[0],
                "slope":       None,
                "abs_delta_f": None,
                "label":       part_type,
                "part_label":  part["label"],
                "part_notes":  part["notes"],
                "part_type":   part_type,
                "points":      list(zip(times, freqs)),
            })
            continue

        part_segs = _runs_to_segments(
            times, freqs,
            abs_threshold_hz=abs_threshold_hz,
            pct_threshold=pct_threshold,
            noise_floor_hz=noise_floor_hz,
            max_merge_duration=max_merge_duration,
        )
        for seg in part_segs:
            seg["part_label"] = part["label"]
            seg["part_notes"] = part["notes"]
            seg["part_type"]  = part_type
        segments.extend(part_segs)

    return segments


def segments_to_token_string(segments: list[dict]) -> str:
    """Collapse consecutive identical labels into a morpheme string.
    E.g. [UP, UP, FLAT, DOWN] -> 'UP-FLAT-DOWN'
    """
    if not segments:
        return ""
    labels = [segments[0]["label"]]
    for seg in segments[1:]:
        if seg["label"] != labels[-1]:
            labels.append(seg["label"])
    return "-".join(labels)


# ── point helpers ─────────────────────────────────────────────────────────────
def _merge_points(segs: list[dict]) -> list[tuple]:
    """Concatenate points from a list of segments, deduplicating shared endpoints."""
    result = []
    for k, s in enumerate(segs):
        pts = s["points"]
        result.extend(pts if k == 0 else pts[1:])
    return result


# ── second-order classification ───────────────────────────────────────────────
def parse_second_order(
    segments: list[dict],
    *,
    flat_tolerance_seconds: float = 0.05,
) -> list[dict]:
    """
    Post-process classify_segments output to identify PEAK and VALLEY shapes.

    PEAK:   UP  [FLAT...] DOWN  — total intervening FLAT duration <= flat_tolerance_seconds
    VALLEY: DOWN [FLAT...] UP   — same tolerance

    Rules:
      - Only 'PRECISE' segments participate.
      - Matches never cross part boundaries.
      - Matched groups are collapsed into one segment; the original sub-segments
        are preserved under 'sub_segments' in case the detail is needed later.
      - Unmatched segments are passed through unchanged.
    """
    result = []
    i = 0
    while i < len(segments):
        seg   = segments[i]
        label = seg["label"]

        if label not in ("UP", "DOWN") or seg.get("part_type") != "PRECISE":
            result.append(seg)
            i += 1
            continue

        target    = "DOWN" if label == "UP" else "UP"
        out_label = "PEAK" if label == "UP" else "VALLEY"
        part      = seg["part_label"]

        # Scan forward past FLAT segments within the same part.
        j = i + 1
        flat_duration = 0.0
        while (
            j < len(segments)
            and segments[j]["label"] == "FLAT"
            and segments[j]["part_label"] == part
        ):
            flat_duration += segments[j]["delta_t"]
            j += 1

        partner_found = (
            j < len(segments)
            and segments[j]["label"] == target
            and segments[j]["part_label"] == part
            and flat_duration <= flat_tolerance_seconds
        )

        if partner_found:
            group = segments[i : j + 1]
            result.append({
                "t_start":      group[0]["t_start"],
                "t_end":        group[-1]["t_end"],
                "f_start":      group[0]["f_start"],
                "f_end":        group[-1]["f_end"],
                "delta_t":      group[-1]["t_end"] - group[0]["t_start"],
                "label":        out_label,
                "part_label":   group[0]["part_label"],
                "part_type":    group[0]["part_type"],
                "part_notes":   group[0]["part_notes"],
                "points":       _merge_points(group),
                "sub_segments": group,
            })
            i = j + 1
        else:
            result.append(seg)
            i += 1

    return result


# ── third-order classification ────────────────────────────────────────────────
def parse_third_order(segments: list[dict]) -> list[dict]:
    """
    Post-process parse_second_order output to identify SQUIGGLE shapes.

    If a part contains more than 2 PEAK/VALLEY segments (counted together),
    the entire part is collapsed into a single SQUIGGLE segment.

    Non-PRECISE parts are passed through unchanged.
    """
    from collections import defaultdict

    by_part: dict[str, list[dict]] = defaultdict(list)
    order = []
    seen = set()
    for seg in segments:
        pl = seg["part_label"]
        by_part[pl].append(seg)
        if pl not in seen:
            order.append(pl)
            seen.add(pl)

    result = []
    for pl in order:
        part_segs = by_part[pl]
        pv_count = sum(1 for s in part_segs if s["label"] in ("PEAK", "VALLEY"))

        if pv_count > 2:
            sub = list(part_segs)
            result.append({
                "t_start":      sub[0]["t_start"],
                "t_end":        sub[-1]["t_end"],
                "f_start":      sub[0]["f_start"],
                "f_end":        sub[-1]["f_end"],
                "delta_t":      sub[-1]["t_end"] - sub[0]["t_start"],
                "label":        "SQUIGGLE",
                "part_label":   pl,
                "part_type":    sub[0]["part_type"],
                "part_notes":   sub[0]["part_notes"],
                "points":       _merge_points(sub),
                "sub_segments": sub,
            })
        else:
            result.extend(part_segs)

    return result
