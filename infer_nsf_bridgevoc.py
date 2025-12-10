import argparse
import os
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

from div.data_module import load_wav, mel_spectrogram
from div.nsf.f0_utils import extract_f0_fcpe
from div.nsf_bridge.score_model import NsfBridgeScoreModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSF-BridgeVoc 推理脚本（wav -> mel + F0[torchfcpe] -> wav）")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="训练好的 NSF-BridgeVoc 检查点路径（Lightning .ckpt）",
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="待重建的输入 wav 路径，将自动用 mel_spectrogram + torchfcpe 提取特征",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="test_decode/nsf_bridgevoc_out.wav",
        help="输出合成 wav 保存路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，默认为 cuda（若可用），否则 cpu",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="可选：覆盖模型 SDE 的采样步数 N（不写则使用训练时的 N）",
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


@torch.no_grad()
def main():
    args = _parse_args()
    device = torch.device(args.device)

    # 1) 从 Lightning ckpt 加载完整的 NSF-Bridge Score 模型（包含 SDE + NsfBcdBridge）
    print(f"[INFO] Loading NsfBridgeScoreModel from {args.ckpt}")
    model = NsfBridgeScoreModel.load_from_checkpoint(
        args.ckpt, map_location=device, strict=False
    )
    model.to(device)
    # 推理场景下：若 ckpt 中没有保存 EMA（老模型），会在 on_load_checkpoint 里自动关闭 EMA；
    # 这里显式传 no_ema=True，确保不会在 eval() 时用随机 shadow_params 覆盖已训练好的 dnn 权重。
    model.eval(no_ema=True)

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

    # 3) 由 mel + F0 构造条件谱 Y，并在复杂谱域做与训练完全一致的谱压缩
    cond_full_ri = model.dnn.build_cond(mel, f0)  # (1, 2, F_full, T)
    if model.drop_last_freq:
        cond_full_ri = cond_full_ri[:, :, :-1].contiguous()  # (1, 2, F-1, T)

    cond_complex = torch.complex(cond_full_ri[:, 0], cond_full_ri[:, 1])  # (1, F-1, T)
    cond_complex = model._spec_fwd(cond_complex)
    cond = torch.stack([cond_complex.real, cond_complex.imag], dim=1)  # (1, 2, F-1, T)

    # 4) BridgeGAN 逆扩散：从条件谱 Y 出发，通过 score 网络恢复目标谱 X
    print(f"[INFO] Running BridgeGAN reverse diffusion with N={model.sde.N}")
    sample_ri = model.sde.reverse_diffusion(
        cond, cond, model.dnn, to_cpu=False
    )  # (1, 2, F-1, T)

    # 5) 与训练相同的反变换流程：压缩谱 -> 原始 STFT -> 波形
    sample_complex = torch.complex(sample_ri[:, 0], sample_ri[:, 1])  # (1, F-1, T)
    if model.drop_last_freq:
        last = sample_complex[:, -1:, :].contiguous()
        sample_complex = torch.cat([sample_complex, last], dim=1)  # (1, F, T)

    spec = model._spec_back(sample_complex)

    target_len = spec.shape[-1] * hop_size
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=torch.hann_window(win_size).to(spec.device),
        center=True,
        length=target_len,
    )
    wav = wav.squeeze().cpu().numpy().astype(np.float32)

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
