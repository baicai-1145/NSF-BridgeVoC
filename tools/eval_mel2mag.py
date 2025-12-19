#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from div.nsf.f0_utils import extract_f0_fcpe
from div.nsf_bridge.mel2mag import Mel2MagConfig, Mel2MagHF


def _read_manifest(path: str) -> Iterable[dict]:
    """
    JSONL: each line is an object like {"wav_path": "...", "name": "..."}.
    Fallback: plain text list (one wav path per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{lineno}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Each JSONL line must be an object at {path}:{lineno}")
                yield obj
            else:
                yield {"wav_path": raw}


def _hz_per_bin(sr: int, n_fft: int) -> float:
    return float(sr) / float(n_fft)


def _hz_to_bin(hz: float, *, sr: int, n_fft: int) -> int:
    return int(round(float(hz) / _hz_per_bin(sr, n_fft)))


def _stft_mag(
    audio: torch.Tensor, *, n_fft: int, hop: int, win: int, window: torch.Tensor, drop_last_freq: bool
) -> torch.Tensor:
    if audio.ndim == 2:
        audio = audio.squeeze(0)
    spec = torch.stft(
        audio,
        n_fft=int(n_fft),
        hop_length=int(hop),
        win_length=int(win),
        window=window.to(audio.device),
        center=True,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-8)
    if drop_last_freq:
        mag = mag[:-1, :].contiguous()
    return mag


def _mel_spectrogram_log(
    audio: torch.Tensor,
    *,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    num_mels: int,
    fmin: float,
    fmax: float,
    window: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mel_log: (n_mels, T)
      mel_basis: (n_mels, F_full)  where F_full = n_fft//2+1
    """
    try:
        from librosa.filters import mel as librosa_mel_fn
    except Exception as e:
        raise ImportError("librosa is required to compute mel spectrogram in eval_mel2mag.py") from e

    if audio.ndim == 2:
        audio = audio.squeeze(0)
    spec = torch.stft(
        audio,
        n_fft=int(n_fft),
        hop_length=int(hop),
        win_length=int(win),
        window=window.to(audio.device),
        center=True,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-8)  # (F, T)
    mel_np = librosa_mel_fn(sr=int(sr), n_fft=int(n_fft), n_mels=int(num_mels), fmin=float(fmin), fmax=float(fmax))
    mel_basis = torch.from_numpy(mel_np).to(audio.device).float()  # (n_mels, F)
    mel_mag = mel_basis @ mag
    mel_log = torch.log(torch.clamp(mel_mag, min=1e-5))
    return mel_log, mel_basis


def _inverse_mel_pinv(mel_log: torch.Tensor, *, mel_basis: torch.Tensor) -> torch.Tensor:
    """
    mel_log: (n_mels, T), log-mel
    mel_basis: (n_mels, F_full)
    return: mag_inv (F_full, T)  linear magnitude (may be negative before abs)
    """
    mel_lin = torch.exp(mel_log).clamp_min(1e-8)
    inv_basis = torch.pinverse(mel_basis)  # (F, n_mels)
    return inv_basis @ mel_lin


def _db(mag: torch.Tensor) -> torch.Tensor:
    mag = mag.clamp_min(1e-7)
    return 20.0 * torch.log10(mag)


def _hf_metrics_from_db(db: torch.Tensor, *, sr: int, n_fft: int, fmin_hz: float, fmax_hz: float) -> dict:
    hz_per_bin = _hz_per_bin(sr, n_fft)
    fmin_bin = int(max(0, round(float(fmin_hz) / hz_per_bin)))
    fmax_bin = int(min(db.shape[0] - 1, round(float(fmax_hz) / hz_per_bin)))
    if fmax_bin <= fmin_bin:
        return {"ok": False, "reason": "invalid_hf_bin_range"}
    hf = db[fmin_bin : fmax_bin + 1]
    q10 = torch.quantile(hf, 0.10).item()
    q50 = torch.quantile(hf, 0.50).item()
    q90 = torch.quantile(hf, 0.90).item()
    q99 = torch.quantile(hf, 0.99).item()
    return {
        "ok": True,
        "hf_bin_min": fmin_bin,
        "hf_bin_max": fmax_bin,
        "hf_q10_db": float(q10),
        "hf_q50_db": float(q50),
        "hf_q90_db": float(q90),
        "hf_q99_db": float(q99),
        "hf_contrast_db": float(q90 - q10),
    }


def _load_mel2mag_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[Mel2MagHF, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {}) or {}
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint format: missing state_dict")

    # Required hyperparams (fallback to common 44.1k settings if missing)
    n_fft = int(hp.get("n_fft", 2048))
    drop_last = bool(hp.get("drop_last_freq", True))
    num_mels = int(hp.get("num_mels", 128))
    hidden = int(hp.get("hidden", 256))
    n_blocks = int(hp.get("n_blocks", 6))
    kernel_size = int(hp.get("kernel_size", 5))
    dropout = float(hp.get("dropout", 0.0))
    f0_max = float(hp.get("f0_max", 1100.0))
    eps = float(hp.get("eps", 1e-6))

    n_freq = (n_fft // 2 + 1) - (1 if drop_last else 0)
    cfg = Mel2MagConfig(
        num_mels=num_mels,
        n_freq=n_freq,
        hidden=hidden,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        f0_max=f0_max,
        eps=eps,
    )
    model = Mel2MagHF(cfg).to(device)
    model.eval()

    # Strip "mel2mag." prefix (Lightning module) if present
    filtered = {}
    for k, v in state.items():
        if k.startswith("mel2mag."):
            filtered[k[len("mel2mag.") :]] = v
        elif k in model.state_dict():
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    meta = {
        "ckpt_path": os.path.abspath(ckpt_path),
        "hyper_parameters": hp,
        "cfg": asdict(cfg),
        "loaded_keys": int(len(filtered)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, meta


def _slice_audio(audio: np.ndarray, sr: int, start_s: float, max_s: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if start_s > 0:
        s0 = int(round(float(start_s) * sr))
        audio = audio[s0:]
    if max_s > 0:
        n = int(round(float(max_s) * sr))
        audio = audio[:n]
    return audio


def _qrange(arr: np.ndarray, q0: float, q1: float) -> Tuple[float, float]:
    lo, hi = np.quantile(arr, [q0, q1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.min(arr)), float(np.max(arr))
    return float(lo), float(hi)


def _save_compare_figure(
    out_png: str,
    *,
    gt_db: np.ndarray,
    inv_db: np.ndarray,
    pr_db: np.ndarray,
    diff_pr_db: np.ndarray,
    title: str,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to render png: {e}") from e

    vmin, vmax = _qrange(gt_db, 0.02, 0.98)
    dmin, dmax = _qrange(diff_pr_db, 0.02, 0.98)
    dlim = max(abs(dmin), abs(dmax), 1.0)

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), constrained_layout=True)
    im0 = axes[0].imshow(gt_db, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("GT STFT |X| (dB)")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)

    im1 = axes[1].imshow(inv_db, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("inverse_mel(pinverse) |M_inv| (dB)")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)

    im2 = axes[2].imshow(pr_db, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[2].set_title("mel2mag |M_hat| (dB)")
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

    im3 = axes[3].imshow(diff_pr_db, origin="lower", aspect="auto", cmap="coolwarm", vmin=-dlim, vmax=dlim)
    axes[3].set_title("diff (mel2mag - GT) (dB)")
    fig.colorbar(im3, ax=axes[3], fraction=0.02, pad=0.01)

    for ax in axes:
        ax.set_ylabel("Freq bin")
    axes[-1].set_xlabel("Frame")
    fig.suptitle(title)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate mel2mag checkpoint against inverse_mel(pinverse) baseline.")
    parser.add_argument("--ckpt", type=str, required=True, help="Mel2MagLightning checkpoint path.")
    parser.add_argument("--wav", type=str, action="append", default=[], help="Input wav path (repeatable).")
    parser.add_argument("--manifest", type=str, default=None, help="Optional JSONL manifest with wav_path/name.")
    parser.add_argument("--out_dir", type=str, default="test_decode/mel2mag_eval", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for mel2mag forward.")
    parser.add_argument("--start_s", type=float, default=0.0, help="Start offset seconds.")
    parser.add_argument("--max_s", type=float, default=12.0, help="Max duration seconds (<=0 means full).")
    parser.add_argument("--sr", type=int, default=44100, help="Sampling rate (must match training).")
    parser.add_argument("--n_fft", type=int, default=2048, help="n_fft (must match training).")
    parser.add_argument("--hop", type=int, default=512, help="hop_size (must match training).")
    parser.add_argument("--win", type=int, default=2048, help="win_size (must match training).")
    parser.add_argument("--num_mels", type=int, default=128, help="num_mels (must match training).")
    parser.add_argument("--fmin", type=float, default=0.0, help="mel fmin.")
    parser.add_argument("--fmax", type=float, default=22050.0, help="mel fmax.")
    parser.add_argument("--drop_last_freq", action="store_true", help="Drop last freq bin to match training F.")
    parser.add_argument("--hf_fmin", type=float, default=6000.0, help="HF band lower bound (Hz) for metrics.")
    parser.add_argument("--hf_fmax", type=float, default=15000.0, help="HF band upper bound (Hz) for metrics.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model, meta = _load_mel2mag_from_ckpt(args.ckpt, device)

    items: List[dict] = []
    if args.manifest:
        for obj in _read_manifest(args.manifest):
            items.append(obj)
    for w in args.wav:
        items.append({"wav_path": w})
    if not items:
        raise SystemExit("No input wav provided. Use --wav or --manifest.")

    results_path = os.path.join(args.out_dir, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", **meta}, ensure_ascii=False) + "\n")

        for it in items:
            wav_path = it.get("wav_path") or it.get("wav") or it.get("path")
            if not wav_path:
                continue
            name = it.get("name") or os.path.splitext(os.path.basename(str(wav_path)))[0]
            out_base = os.path.join(args.out_dir, name)
            out_npz = out_base + ".npz"
            out_png = out_base + ".png"

            audio, sr0 = sf.read(wav_path, dtype="float32", always_2d=False)
            if sr0 != int(args.sr):
                raise ValueError(f"sr mismatch: wav sr={sr0}, expected {int(args.sr)} for {wav_path}")
            audio = _slice_audio(audio, int(args.sr), float(args.start_s), float(args.max_s))
            audio = np.clip(audio, -0.95, 0.95)

            audio_t = torch.from_numpy(audio).to(device).float()
            window = torch.hann_window(int(args.win), device=device)

            # GT mag & mel
            gt_mag = _stft_mag(
                audio_t,
                n_fft=int(args.n_fft),
                hop=int(args.hop),
                win=int(args.win),
                window=window,
                drop_last_freq=bool(args.drop_last_freq),
            )  # (F,T)
            mel_log, mel_basis = _mel_spectrogram_log(
                audio_t,
                sr=int(args.sr),
                n_fft=int(args.n_fft),
                hop=int(args.hop),
                win=int(args.win),
                num_mels=int(args.num_mels),
                fmin=float(args.fmin),
                fmax=float(args.fmax),
                window=window,
            )  # (n_mels,Tm)

            frames = int(mel_log.shape[-1])
            f0 = extract_f0_fcpe(
                audio,
                sr=int(args.sr),
                n_frames=frames,
                device=str(device),
                f0_min=80.0,
                f0_max=1100.0,
                threshold=0.006,
                interp_uv=False,
            )
            f0_t = torch.from_numpy(f0).to(device).float().unsqueeze(0)  # (1,T)
            mel_t = mel_log.unsqueeze(0).to(device)  # (1,n_mels,T)
            uv_t = (f0_t > 0).to(mel_t.dtype)

            with torch.no_grad():
                pr_mag = model(mel_t, f0_t, uv_t)[0]  # (F,T)

            inv_mag_raw = _inverse_mel_pinv(mel_log, mel_basis=mel_basis).abs().clamp_min(1e-8)  # (F_full,T)
            if bool(args.drop_last_freq):
                inv_mag = inv_mag_raw[:-1, :].contiguous()
            else:
                inv_mag = inv_mag_raw

            # Align
            F_common = min(int(gt_mag.shape[0]), int(pr_mag.shape[0]), int(inv_mag.shape[0]))
            T_common = min(int(gt_mag.shape[1]), int(pr_mag.shape[1]), int(inv_mag.shape[1]))
            gt_mag = gt_mag[:F_common, :T_common]
            pr_mag = pr_mag[:F_common, :T_common]
            inv_mag = inv_mag[:F_common, :T_common]

            gt_db = _db(gt_mag).detach().cpu()
            pr_db = _db(pr_mag).detach().cpu()
            inv_db = _db(inv_mag).detach().cpu()

            hf_gt = _hf_metrics_from_db(gt_db, sr=int(args.sr), n_fft=int(args.n_fft), fmin_hz=float(args.hf_fmin), fmax_hz=float(args.hf_fmax))
            hf_pr = _hf_metrics_from_db(pr_db, sr=int(args.sr), n_fft=int(args.n_fft), fmin_hz=float(args.hf_fmin), fmax_hz=float(args.hf_fmax))
            hf_inv = _hf_metrics_from_db(inv_db, sr=int(args.sr), n_fft=int(args.n_fft), fmin_hz=float(args.hf_fmin), fmax_hz=float(args.hf_fmax))

            # L1 in HF band (dB)
            hf_l1_pr = float("nan")
            hf_l1_inv = float("nan")
            if hf_gt.get("ok") and hf_pr.get("ok"):
                b0, b1 = int(hf_gt["hf_bin_min"]), int(hf_gt["hf_bin_max"])
                hf_l1_pr = float(torch.mean(torch.abs(pr_db[b0 : b1 + 1] - gt_db[b0 : b1 + 1])).item())
            if hf_gt.get("ok") and hf_inv.get("ok"):
                b0, b1 = int(hf_gt["hf_bin_min"]), int(hf_gt["hf_bin_max"])
                hf_l1_inv = float(torch.mean(torch.abs(inv_db[b0 : b1 + 1] - gt_db[b0 : b1 + 1])).item())

            metrics = {
                "hf_l1_db_pr": hf_l1_pr,
                "hf_l1_db_inv": hf_l1_inv,
                "hf_contrast_delta_pr": (float(hf_pr.get("hf_contrast_db", np.nan)) - float(hf_gt.get("hf_contrast_db", np.nan))) if hf_gt.get("ok") and hf_pr.get("ok") else float("nan"),
                "hf_contrast_delta_inv": (float(hf_inv.get("hf_contrast_db", np.nan)) - float(hf_gt.get("hf_contrast_db", np.nan))) if hf_gt.get("ok") and hf_inv.get("ok") else float("nan"),
                "hf_floor_delta_pr": (float(hf_pr.get("hf_q50_db", np.nan)) - float(hf_gt.get("hf_q50_db", np.nan))) if hf_gt.get("ok") and hf_pr.get("ok") else float("nan"),
                "hf_floor_delta_inv": (float(hf_inv.get("hf_q50_db", np.nan)) - float(hf_gt.get("hf_q50_db", np.nan))) if hf_gt.get("ok") and hf_inv.get("ok") else float("nan"),
            }

            # Save arrays
            np.savez_compressed(
                out_npz,
                gt_db=gt_db.numpy().astype(np.float32),
                inv_db=inv_db.numpy().astype(np.float32),
                pr_db=pr_db.numpy().astype(np.float32),
                diff_pr_db=(pr_db - gt_db).numpy().astype(np.float32),
                diff_inv_db=(inv_db - gt_db).numpy().astype(np.float32),
                sampling_rate=np.int32(int(args.sr)),
                n_fft=np.int32(int(args.n_fft)),
                hop=np.int32(int(args.hop)),
                win=np.int32(int(args.win)),
                drop_last_freq=bool(args.drop_last_freq),
                hf_fmin_hz=np.float32(float(args.hf_fmin)),
                hf_fmax_hz=np.float32(float(args.hf_fmax)),
                metrics=metrics,
            )

            title = f"{name} | hf_l1_db inv={hf_l1_inv:.3f} pr={hf_l1_pr:.3f} | floorΔ inv={metrics['hf_floor_delta_inv']:+.2f} pr={metrics['hf_floor_delta_pr']:+.2f}"
            try:
                _save_compare_figure(
                    out_png,
                    gt_db=gt_db.numpy(),
                    inv_db=inv_db.numpy(),
                    pr_db=pr_db.numpy(),
                    diff_pr_db=(pr_db - gt_db).numpy(),
                    title=title,
                )
            except Exception as e:
                print(f"[WARN] render png failed for {name}: {e}. Arrays saved to: {out_npz}")

            row = {
                "type": "item",
                "name": name,
                "wav_path": os.path.abspath(str(wav_path)),
                "out_npz": os.path.abspath(out_npz),
                "out_png": os.path.abspath(out_png) if os.path.exists(out_png) else None,
                "metrics": metrics,
                "hf_gt": hf_gt,
                "hf_inv": hf_inv,
                "hf_pr": hf_pr,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[OK] {name}: hf_l1_db inv={hf_l1_inv:.3f} pr={hf_l1_pr:.3f}  saved={out_npz}")

    print(f"[INFO] Done. results.jsonl: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    main()

