"""Batch contour editor: iterate files from processed .txt labels or directly from audio.

Usage:
    # From processed folder (.txt label files):
    uv run annotation_audacity/contour/batch_editor.py --processed-folder data/ford-catalogue/audacity_maddie/processed --audio-folder "data/ford-catalogue/Northern Resident/media/G_clan"
    uv run annotation_audacity/contour/batch_editor.py --save-folder annotations/ --skip-done --audio-folder "data/ford-catalogue/Northern Resident/media/A_clan"
    uv run annotation_audacity/contour/batch_editor.py --save-folder annotations/ --start N07i-A1-1

    # From audio folder only (.wav/.mp3 files, no processed folder):
    uv run annotation_audacity/contour/batch_editor.py --audio-folder path/to/audio

Controls:
    Save      — save current file (stay on it)
    Next →    — save + advance to next file
    Close window manually — advance without saving
"""
import os
import argparse
import matplotlib
matplotlib.use("TkAgg")  # ensure interactive backend

from contour_editor import run_editor, load_saved_annotations


def collect_filenames_from_txt(processed_folder: str) -> list[str]:
    """Return sorted list of base filenames from .txt files in processed_folder."""
    names = []
    for f in sorted(os.listdir(processed_folder)):
        if f.endswith(".txt"):
            names.append(os.path.splitext(f)[0])
    return names


def collect_filenames_from_audio(audio_folder: str) -> list[str]:
    """Return sorted list of base filenames from .wav and .mp3 files in audio_folder."""
    names = set()
    for f in os.listdir(audio_folder):
        if f.endswith(".wav") or f.endswith(".mp3"):
            names.add(os.path.splitext(f)[0])
    return sorted(names)


def run_batch(
    audio_folder: str,
    processed_folder: str | None,
    save_folder: str,
    spects_folder: str,
    n_control_points: int,
    n_harmonics: int,
    n_fft: int,
    hop_length: int,
    skip_done: bool,
    start_at: str | None,
):
    if processed_folder:
        filenames = collect_filenames_from_txt(processed_folder)
        if not filenames:
            print(f"No .txt files found in {processed_folder}")
            return
    else:
        filenames = collect_filenames_from_audio(audio_folder)
        if not filenames:
            print(f"No .wav or .mp3 files found in {audio_folder}")
            return

    # Optionally skip already-annotated files
    if skip_done:
        before = len(filenames)
        filenames = [
            fn for fn in filenames
            if load_saved_annotations(save_folder, fn) is None
        ]
        print(f"Skipping {before - len(filenames)} already-annotated files.")

    # Optionally start from a specific file
    if start_at is not None:
        try:
            idx = filenames.index(start_at)
            filenames = filenames[idx:]
        except ValueError:
            print(f"Warning: --start '{start_at}' not found in file list; starting from the beginning.")

    total = len(filenames)
    if total == 0:
        print("Nothing to annotate.")
        return

    print(f"Batch: {total} file(s) to annotate.")

    i = 0
    while i < total:
        filename = filenames[i]
        print(f"\n[{i + 1}/{total}] {filename}")

        # Check if audio file exists in the audio folder
        audio_exists = any(
            os.path.exists(os.path.join(audio_folder, f"{filename}{ext}"))
            for ext in (".mp3", ".wav")
        )
        if not audio_exists:
            print(f"  Audio not found in {audio_folder} — skipping.")
            i += 1
            continue

        advance = [False]

        def on_next():
            advance[0] = True

        try:
            run_editor(
                folder_path=audio_folder,
                filename=filename,
                n_control_points=n_control_points,
                n_harmonics=n_harmonics,
                save_folder=save_folder,
                spects_folder=spects_folder,
                n_fft=n_fft,
                hop_length=hop_length,
                on_next=on_next,
            )
        except Exception as exc:
            print(f"  Error on {filename}: {exc}")

        i += 1

    print("\nBatch complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch contour editor")
    parser.add_argument(
        "--audio-folder",
        type=str,
        default="data/ford-catalogue/Northern Resident/media/A_clan",
        help="Folder containing audio + .f0.csv files",
    )
    parser.add_argument(
        "--processed-folder",
        type=str,
        default=None,
        help="Folder with Audacity .txt label files (source of filenames). If omitted, use .wav/.mp3 from --audio-folder",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="annotations/",
        help="Folder for saved JSON annotations",
    )
    parser.add_argument(
        "--spects-folder",
        type=str,
        default="data/ford_paper_spects",
        help="Folder with *_paper_spect.png reference images",
    )
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip files that already have a saved JSON in --save-folder",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Start batch from this filename (e.g. N07i-A1-1)",
    )
    args = parser.parse_args()

    run_batch(
        audio_folder=args.audio_folder,
        processed_folder=args.processed_folder,
        save_folder=args.save_folder,
        spects_folder=args.spects_folder,
        n_control_points=args.points,
        n_harmonics=args.harmonics,
        n_fft=args.nfft,
        hop_length=args.hop,
        skip_done=args.skip_done,
        start_at=args.start,
    )
