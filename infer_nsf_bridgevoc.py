import argparse
import os
import random
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

from div.data_module import load_wav, mel_spectrogram
from div.nsf.f0_utils import extract_f0_fcpe
from div.nsf_bridge.score_model import NsfBridgeScoreModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSF-BridgeVoc inference script (wav -> mel + F0[torchfcpe] -> wav)")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained NSF-BridgeVoc checkpoint (.ckpt from Lightning).",
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="Input wav path; mel_spectrogram and torchfcpe features will be computed automatically.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="test_decode/nsf_bridgevoc_out.wav",
        help="Output path to save the synthesized wav.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, defaults to cuda when available.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Optional: override the model SDE sampling steps N (default uses training N).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help="Optional: number of mel frames per chunk for memory-friendly inference.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Optional: chunk length in seconds (ignored if --chunk-frames is set).",
    )
    parser.add_argument(
        "--crossfade-seconds",
        type=float,
        default=0.12,
        help="Optional: cross-fade duration in seconds when stitching chunked outputs; set 0 to disable overlap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional: set a random seed for deterministic reverse diffusion sampling.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Disable EMA weights for inference (default uses EMA if available in the checkpoint).",
    )
    return parser.parse_args()


def _extract_mel_f0_from_wav(
    wav_path: str,
    sampling_rate: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    num_mels: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    从单个 wav 中提取 mel 频谱与 F0（使用 torchfcpe），返回:
      mel: (1, num_mels, frames)
      f0:  (1, frames)
    """
    # 读入并裁剪到 [-1, 1]
    audio = load_wav(wav_path, sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -0.95, 0.95)

    # mel 计算与训练/预处理保持一致
    audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    mel = mel_spectrogram(
        audio_t,
        n_fft=n_fft,
        num_mels=num_mels,
        sampling_rate=sampling_rate,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        center=True,
        in_dataset=False,
    )[0]  # (num_mels, frames)

    frames = mel.shape[1]
    target_len = frames * hop_size
    if audio.shape[0] < target_len:
        pad = target_len - audio.shape[0]
        audio = np.pad(audio, (0, pad), mode="constant")
    else:
        audio = audio[:target_len]

    # 记录与模型输入对齐后的参考峰值，用于推理后响度匹配
    ref_peak = float(np.max(np.abs(audio)) + 1e-8)

    # 使用 torchfcpe 提取 F0，并对齐到 frames
    f0 = extract_f0_fcpe(
        audio,
        sr=sampling_rate,
        n_frames=frames,
        device=device,
        f0_min=80.0,
        f0_max=880.0,
        threshold=0.006,
        interp_uv=False,
    )

    mel = mel.unsqueeze(0)  # (1, num_mels, frames)
    f0_t = torch.from_numpy(f0).unsqueeze(0)  # (1, frames)
    return mel, f0_t, ref_peak


def _build_cond(model: NsfBridgeScoreModel, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """Build conditioning features for the score model."""
    cond_full_ri = model.dnn.build_cond(mel, f0)  # (B, 2, F_full, T)
    if model.drop_last_freq:
        cond_full_ri = cond_full_ri[:, :, :-1].contiguous()  # (B, 2, F-1, T)

    cond_complex = torch.complex(cond_full_ri[:, 0], cond_full_ri[:, 1])  # (B, F-1, T)
    cond_complex = model._spec_fwd(cond_complex)
    cond = torch.stack([cond_complex.real, cond_complex.imag], dim=1)  # (B, 2, F-1, T)
    return cond


def _reverse_to_wav(
    model: NsfBridgeScoreModel,
    cond: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    window: torch.Tensor,
) -> np.ndarray:
    """Run reverse diffusion on a single chunk and convert to waveform."""
    sample_ri = model.sde.reverse_diffusion(cond, cond, model.dnn, to_cpu=False)  # (B, 2, F-1, T)

    sample_complex = torch.complex(sample_ri[:, 0], sample_ri[:, 1])  # (B, F-1, T)
    if model.drop_last_freq:
        # onesided STFT 的 Nyquist bin：推理时同样用 0 填充，避免复制相邻频带把能量带到 Nyquist 附近。
        last = torch.zeros_like(sample_complex[:, :1, :]).contiguous()
        sample_complex = torch.cat([sample_complex, last], dim=1)  # (B, F, T)

    spec = model._spec_back(sample_complex)

    target_len = spec.shape[-1] * hop_size
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        length=target_len,
    )
    return wav.squeeze().cpu().numpy().astype(np.float32)


def _crossfade_concat(accum: np.ndarray, chunk: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """Append chunk to accum with linear cross-fade over the overlap."""
    if accum.size == 0:
        return chunk
    if crossfade_samples <= 0:
        return np.concatenate([accum, chunk])

    overlap = min(crossfade_samples, accum.shape[0], chunk.shape[0])
    if overlap == 0:
        return np.concatenate([accum, chunk])

    fade_out = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    mixed = accum[-overlap:] * fade_out + chunk[:overlap] * fade_in
    return np.concatenate([accum[:-overlap], mixed, chunk[overlap:]])


def _chunked_infer(
    model: NsfBridgeScoreModel,
    mel: torch.Tensor,
    f0: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    chunk_frames: int,
    overlap_frames: int,
) -> np.ndarray:
    """Run memory-friendly inference by chunking along the time dimension."""
    total_frames = mel.shape[-1]
    overlap_frames = max(0, min(overlap_frames, chunk_frames - 1))
    step = max(1, chunk_frames - overlap_frames)
    window = torch.hann_window(win_size, device=mel.device)

    wav_out = np.zeros(0, dtype=np.float32)
    start = 0
    chunk_idx = 0

    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        mel_chunk = mel[..., start:end].contiguous()
        f0_chunk = f0[..., start:end].contiguous()

        chunk_idx += 1
        print(
            f"[INFO] Decoding chunk {chunk_idx}: frames {start} - {end} / {total_frames} "
            f"(len={end - start}, overlap={overlap_frames})"
        )

        cond_chunk = _build_cond(model, mel_chunk, f0_chunk)
        wav_chunk = _reverse_to_wav(model, cond_chunk, n_fft, hop_size, win_size, window)

        wav_out = _crossfade_concat(wav_out, wav_chunk, overlap_frames * hop_size)
        start = end if end == total_frames else start + step

        # Free cached blocks between chunks to mitigate OOM on small GPUs.
        if torch.cuda.is_available() and mel.device.type == "cuda":
            torch.cuda.empty_cache()

    return wav_out


@torch.no_grad()
def main():
    args = _parse_args()
    device = torch.device(args.device)

    # Optional deterministic sampling for reproducible comparisons / deployment.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Set random seed to {args.seed} for deterministic sampling.")

    # 1) 从 Lightning ckpt 加载完整的 NSF-Bridge Score 模型（包含 SDE + NsfBcdBridge）
    print(f"[INFO] Loading NsfBridgeScoreModel from {args.ckpt}")
    model = NsfBridgeScoreModel.load_from_checkpoint(
        args.ckpt, map_location=device, strict=False
    )
    model.to(device)
    # 推理场景下：若 ckpt 中没有保存 EMA（老模型），会在 on_load_checkpoint 里自动关闭 EMA；
    # 这里显式传 no_ema=True，确保不会在 eval() 时用随机 shadow_params 覆盖已训练好的 dnn 权重。
    # By default, use EMA weights if available in the checkpoint for cleaner audio.
    # For older checkpoints without EMA, NsfBridgeScoreModel will automatically disable EMA.
    model.eval(no_ema=args.no_ema)

    # 如有需要，允许通过命令行覆盖采样步数 N
    if args.N is not None:
        print(f"[INFO] Override SDE.N: {model.sde.N} -> {args.N}")
        model.sde.N = int(args.N)

    # 2) 从 wav 提取 mel + F0（torchfcpe）
    sr = model.sampling_rate
    n_fft = model.n_fft
    hop_size = model.hop_size
    win_size = model.win_size
    fmin = model.fmin
    fmax = model.fmax
    num_mels = model.num_mels

    mel, f0, ref_peak = _extract_mel_f0_from_wav(
        args.wav,
        sampling_rate=sr,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        num_mels=num_mels,
        device=str(device),
    )
    mel = mel.to(device)
    f0 = f0.to(device)
    print(f"[INFO] mel shape={mel.shape}, f0 shape={f0.shape}")

    # 3) 选择全长或分片推理，分片时用交叉淡化拼接，缓解小显存 OOM
    window = torch.hann_window(win_size, device=device)

    chunk_frames = None
    if args.chunk_frames is not None and args.chunk_frames > 0:
        chunk_frames = int(args.chunk_frames)
    elif args.chunk_seconds is not None and args.chunk_seconds > 0:
        chunk_frames = int(round(args.chunk_seconds * sr / hop_size))

    if chunk_frames is not None and chunk_frames > 0 and chunk_frames < mel.shape[-1]:
        overlap_frames = int(round(max(args.crossfade_seconds, 0.0) * sr / hop_size))
        overlap_frames = max(0, min(overlap_frames, chunk_frames - 1))
        if chunk_frames - overlap_frames <= 0:
            overlap_frames = max(0, chunk_frames - 1)

        print(
            "[INFO] Chunked inference enabled: "
            f"chunk_frames={chunk_frames}, overlap_frames={overlap_frames} "
            f"(~{chunk_frames * hop_size / sr:.2f}s chunk, ~{overlap_frames * hop_size / sr:.2f}s overlap)"
        )
        wav = _chunked_infer(
            model,
            mel,
            f0,
            n_fft,
            hop_size,
            win_size,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
        )
    else:
        if chunk_frames is not None and chunk_frames >= mel.shape[-1]:
            print("[WARN] chunk size >= total length, falling back to single-shot inference.")

        cond = _build_cond(model, mel, f0)
        print(f"[INFO] Running BridgeGAN reverse diffusion with N={model.sde.N}")
        wav = _reverse_to_wav(model, cond, n_fft, hop_size, win_size, window)

    # 6) 响度匹配与限幅：
    #    - 将模型输出的峰值对齐到输入片段的峰值；
    #    - 同时限制最大幅度不超过 0.95，避免>0 dBFS 的硬剪裁。
    out_peak = float(np.max(np.abs(wav)) + 1e-8)
    if out_peak > 0.0:
        if ref_peak > 0.0:
            target_peak = min(ref_peak, 0.95)
        else:
            target_peak = 0.95
        wav = wav * (target_peak / out_peak)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(args.out, wav, samplerate=sr)
    print(f"[INFO] Saved NSF-BridgeVoc output to: {args.out}")


if __name__ == "__main__":
    main()
