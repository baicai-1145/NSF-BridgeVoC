import os
import glob
import random
import argparse


def build_file_lists(root_dir: str, train_out: str, val_out: str, val_ratio: float = 0.02) -> None:
    wav_paths = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
    wav_paths = [p for p in wav_paths if os.path.isfile(p)]

    if not wav_paths:
        raise RuntimeError(f"No .wav files found under '{root_dir}'.")

    random.seed(3407)
    random.shuffle(wav_paths)

    n_val = max(1, int(len(wav_paths) * val_ratio))
    val_paths = wav_paths[:n_val]
    train_paths = wav_paths[n_val:]

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(val_out), exist_ok=True)

    def write_list(paths, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            for p in paths:
                rel = os.path.relpath(p, root_dir)
                rel_no_ext = os.path.splitext(rel)[0].replace("\\", "/")
                f.write(f"{rel_no_ext}|dummy\n")

    write_list(train_paths, train_out)
    write_list(val_paths, val_out)

    print(f"Total wav: {len(wav_paths)}, train: {len(train_paths)}, val: {len(val_paths)}")
    print(f"Train list written to: {train_out}")
    print(f"Val list written to:   {val_out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a wav directory and generate LibriTTS-style train/val scp files."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/train/audio",
        help="Root directory containing wav files (default: data/train/audio).",
    )
    parser.add_argument(
        "--train_out",
        type=str,
        default="Datascp/LibriTTS/train-full.txt",
        help="Output path for training list (default: Datascp/LibriTTS/train-full.txt).",
    )
    parser.add_argument(
        "--val_out",
        type=str,
        default="Datascp/LibriTTS/val-full.txt",
        help="Output path for validation list (default: Datascp/LibriTTS/val-full.txt).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.02,
        help="Fraction of files used for validation (default: 0.02).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_file_lists(
        root_dir=args.root_dir,
        train_out=args.train_out,
        val_out=args.val_out,
        val_ratio=args.val_ratio,
    )

