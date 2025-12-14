import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

# Allow running as `python tools/debug_phase_mask.py ...` without setting PYTHONPATH.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from div.nsf.nsf_hifigan import SourceModuleHnNSF
from div.data_module import inverse_mel, mel_spectrogram


def _load_wav(path: str, sr: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    return np.clip(audio, -0.95, 0.95)


def _maybe_extract_f0_fcpe(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    device: str,
    f0_min: float,
    f0_max: float,
    threshold: float,
    interp_uv: bool,
) -> np.ndarray:
    try:
        from div.nsf.f0_utils import extract_f0_fcpe
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "未提供 --f0-npy，且无法导入 torchfcpe 的提取函数。"
            "请安装 torchfcpe 或提供预计算的 f0.npy。"
        ) from e

    return extract_f0_fcpe(
        audio,
        sr=sr,
        n_frames=n_frames,
        device=device,
        f0_min=f0_min,
        f0_max=f0_max,
        threshold=threshold,
        interp_uv=interp_uv,
    )


def _pad_or_crop_1d(x: np.ndarray, length: int, pad_mode: str = "edge") -> np.ndarray:
    if x.shape[0] >= length:
        return x[:length].astype(np.float32)
    pad = length - x.shape[0]
    return np.pad(x.astype(np.float32), (0, pad), mode=pad_mode)


@torch.no_grad()
def _compute_mask_stats(
    wav_path: str,
    f0_npy: Optional[str],
    device: torch.device,
    sr: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    num_mels: int,
    harmonic_num: int,
    sine_amp: float,
    add_noise_std: float,
    voiced_threshold: float,
    phase_mask_ratio: float,
    mel_phase_gate_ratio: float,
    f0_min: float,
    f0_max: float,
    fcpe_threshold: float,
    fcpe_interp_uv: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    audio = _load_wav(wav_path, sr=sr)
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)

    mel = mel_spectrogram(
        audio_t,
        n_fft=n_fft,
        num_mels=num_mels,
        sampling_rate=sr,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        center=True,
        in_dataset=False,
    )  # (1, n_mels, frames)
    frames = int(mel.shape[-1])

    target_len = frames * hop_size
    if audio.shape[0] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[0]), mode="constant")
    else:
        audio = audio[:target_len]

    if f0_npy is not None:
        f0_np = np.load(f0_npy).astype(np.float32)
        f0_np = _pad_or_crop_1d(f0_np, frames, pad_mode="edge")
    else:
        f0_np = _maybe_extract_f0_fcpe(
            audio=audio,
            sr=sr,
            n_frames=frames,
            device=str(device),
            f0_min=f0_min,
            f0_max=f0_max,
            threshold=fcpe_threshold,
            interp_uv=fcpe_interp_uv,
        )

    f0 = torch.from_numpy(f0_np).unsqueeze(0).to(device)  # (1, frames)

    source = SourceModuleHnNSF(
        sampling_rate=sr,
        harmonic_num=harmonic_num,
        sine_amp=sine_amp,
        add_noise_std=add_noise_std,
        voiced_threshold=voiced_threshold,
    ).to(device)

    har_source = source(f0, hop_size).transpose(1, 2)  # (1, 1, T)
    window = torch.hann_window(win_size, device=device)
    spec_src = torch.stft(
        har_source.squeeze(1),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        return_complex=True,
    )  # (1, F, T_src)
    mag_src = spec_src.abs()

    mag_mel = inverse_mel(
        mel,
        n_fft=n_fft,
        num_mels=num_mels,
        sampling_rate=sr,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        in_dataset=False,
    ).abs().clamp_min_(1e-6)  # (1, F, T_mel)

    T_common = min(int(mag_src.shape[-1]), int(mag_mel.shape[-1]), int(f0.shape[-1]))
    mag_src = mag_src[..., :T_common]
    mag_mel = mag_mel[..., :T_common]
    f0 = f0[..., :T_common]

    if phase_mask_ratio > 0.0:
        max_mag = mag_src.amax(dim=-2, keepdim=True).clamp_min(1e-8)  # (1, 1, T)
        pha_mask = mag_src > (phase_mask_ratio * max_mag)
    else:
        pha_mask = torch.ones_like(mag_src, dtype=torch.bool)

    uv = (f0 > voiced_threshold).to(mag_src.dtype).unsqueeze(1)  # (1, 1, T)
    pha_mask = pha_mask & (uv > 0.5)

    if mel_phase_gate_ratio > 0.0:
        max_mel = mag_mel.amax(dim=-2, keepdim=True).clamp_min(1e-8)  # (1, 1, T)
        mel_gate = mag_mel > (mel_phase_gate_ratio * max_mel)
        pha_mask = pha_mask & mel_gate
    return mag_mel, mag_src, pha_mask, uv


def _plot_overlay(
    mag_mel: torch.Tensor,
    pha_mask: torch.Tensor,
    out_png: str,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("未安装 matplotlib，无法生成可视化。可加 --no-plot 只打印统计。") from e

    mag = mag_mel[0].detach().float().cpu().numpy()
    mask = pha_mask[0].detach().cpu().numpy().astype(np.float32)
    mag_db = np.log10(np.clip(mag, 1e-7, None))

    plt.figure(figsize=(12, 5))
    plt.imshow(mag_db, origin="lower", aspect="auto", cmap="magma")
    plt.imshow(mask, origin="lower", aspect="auto", cmap="Reds", alpha=0.25, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Freq bins (STFT)")
    plt.colorbar(label="log10(mag_mel)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_mag_compare(
    mag_true: torch.Tensor,
    mag_mel: torch.Tensor,
    out_png: str,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("未安装 matplotlib，无法生成可视化。") from e

    true_ = mag_true[0].detach().float().cpu().numpy()
    mel_ = mag_mel[0].detach().float().cpu().numpy()
    true_db = np.log10(np.clip(true_, 1e-7, None))
    mel_db = np.log10(np.clip(mel_, 1e-7, None))
    diff_db = mel_db - true_db

    vmin = float(np.percentile(true_db, 1))
    vmax = float(np.percentile(true_db, 99))
    dv = float(np.percentile(np.abs(diff_db), 99))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)
    axes[0].imshow(true_db, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("True STFT | log10(mag)")
    axes[1].imshow(mel_db, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("inverse_mel(mel) | log10(mag_mel)")
    axes[2].imshow(diff_db, origin="lower", aspect="auto", cmap="coolwarm", vmin=-dv, vmax=dv)
    axes[2].set_title("diff = log10(mag_mel) - log10(true_mag)")
    axes[2].set_xlabel("Frames")
    for ax in axes:
        ax.set_ylabel("Freq bins")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug NSF phase-mask coverage and visualize overlay on mag_mel.")
    parser.add_argument("wav_pos", nargs="?", help="输入 wav 路径（位置参数，可替代 --wav）")
    parser.add_argument("--wav", type=str, default=None, help="输入 wav 路径")
    parser.add_argument("--f0-npy", type=str, default=None, help="可选：预计算 f0.npy（长度为 mel 帧数）")
    parser.add_argument("--out", type=str, default="debug_phase_mask.png", help="输出图像路径")
    parser.add_argument(
        "--compare-mag-out",
        type=str,
        default=None,
        help="可选：输出 true STFT mag vs inverse_mel(mel) 的对比图 PNG（用于判断 mel 是否已抹平高频谐波）。",
    )
    parser.add_argument("--no-plot", action="store_true", help="只打印统计，不输出图像")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 必须与训练/推理 ckpt 超参一致
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--win-size", type=int, default=2048)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=22050.0)
    parser.add_argument("--num-mels", type=int, default=128)

    # NSF / mask 参数
    parser.add_argument("--harmonic-num", type=int, default=8)
    parser.add_argument("--sine-amp", type=float, default=0.1)
    parser.add_argument("--add-noise-std", type=float, default=0.003)
    parser.add_argument("--voiced-threshold", type=float, default=0.0)
    parser.add_argument("--phase-mask-ratio", type=float, default=0.1)
    parser.add_argument(
        "--mel-phase-gate-ratio",
        type=float,
        default=0.0,
        help="可选：再用 mag_mel 做门控，避免在幅度很低处注入相位（建议 0.005~0.02）。",
    )

    # FCPE（仅在未提供 f0_npy 时使用）
    parser.add_argument("--f0-min", type=float, default=80.0)
    parser.add_argument("--f0-max", type=float, default=880.0)
    parser.add_argument("--fcpe-threshold", type=float, default=0.006)
    parser.add_argument("--fcpe-interp-uv", action="store_true", help="让 torchfcpe 插值 unvoiced（默认关闭）")

    args = parser.parse_args()
    wav_path = args.wav or args.wav_pos
    if not wav_path:
        parser.error("必须提供输入 wav：使用位置参数或 --wav。")

    device = torch.device(args.device)
    mag_mel, mag_src, pha_mask, uv = _compute_mask_stats(
        wav_path=wav_path,
        f0_npy=args.f0_npy,
        device=device,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_size=args.hop_size,
        win_size=args.win_size,
        fmin=args.fmin,
        fmax=args.fmax,
        num_mels=args.num_mels,
        harmonic_num=args.harmonic_num,
        sine_amp=args.sine_amp,
        add_noise_std=args.add_noise_std,
        voiced_threshold=args.voiced_threshold,
        phase_mask_ratio=args.phase_mask_ratio,
        mel_phase_gate_ratio=args.mel_phase_gate_ratio,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        fcpe_threshold=args.fcpe_threshold,
        fcpe_interp_uv=bool(args.fcpe_interp_uv),
    )

    mask_mean = float(pha_mask.float().mean().item())
    voiced_ratio = float(uv.float().mean().item())
    if voiced_ratio > 0:
        mask_voiced_mean = float((pha_mask.float() * uv).sum().item() / (uv.sum().item() * pha_mask.shape[-2]))
    else:
        mask_voiced_mean = 0.0

    print(f"[pha_mask] mean={mask_mean:.6f}")
    print(f"[uv] voiced_ratio={voiced_ratio:.6f}")
    print(f"[pha_mask|voiced] mean={mask_voiced_mean:.6f}")
    print(f"mag_mel shape={tuple(mag_mel.shape)}, mag_src shape={tuple(mag_src.shape)}")

    if not args.no_plot:
        title = (
            f"phase_mask_ratio={args.phase_mask_ratio}, voiced_threshold={args.voiced_threshold}, "
            f"add_noise_std={args.add_noise_std}"
        )
        _plot_overlay(mag_mel=mag_mel, pha_mask=pha_mask, out_png=args.out, title=title)
        print(f"[INFO] saved overlay png: {args.out}")

    if args.compare_mag_out is not None:
        # Compute true STFT magnitude from the input wav aligned to mag_mel shape.
        # Note: this is only for visualization/diagnosis, not part of training/inference.
        audio = _load_wav(wav_path, sr=args.sr)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        window = torch.hann_window(args.win_size, device=device)
        spec_true = torch.stft(
            audio_t,
            args.n_fft,
            hop_length=args.hop_size,
            win_length=args.win_size,
            window=window,
            center=True,
            return_complex=True,
        )
        mag_true = spec_true.abs()
        T_common = min(mag_true.shape[-1], mag_mel.shape[-1])
        mag_true = mag_true[..., :T_common]
        mag_mel_c = mag_mel[..., :T_common]
        cmp_title = f"sr={args.sr}, n_fft={args.n_fft}, hop={args.hop_size}, win={args.win_size}"
        _plot_mag_compare(mag_true=mag_true, mag_mel=mag_mel_c, out_png=args.compare_mag_out, title=cmp_title)
        print(f"[INFO] saved mag compare png: {args.compare_mag_out}")


if __name__ == "__main__":
    main()
