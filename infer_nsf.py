import argparse
import os
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

from div.data_module import load_wav, mel_spectrogram
from div.nsf.f0_utils import extract_f0_fcpe
from div.nsf.module import NsfHifiGanModel

try:
    import yaml
except ImportError:
    yaml = None


def parse_args():
    parser = argparse.ArgumentParser(description="NSF-HiFiGAN inference for NSF-BridgeVoC")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练时使用的 YAML 配置，例如 configs/nsf_hifigan_44k1.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="训练好的 NSF-HiFiGAN 检查点路径（.ckpt）",
    )
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="待重建的输入 wav 路径（会用同样的 mel+F0 管线提取特征）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="test_decode/nsf_out.wav",
        help="输出合成 wav 保存路径",
    )
    return parser.parse_args()


def load_config(path: str):
    if yaml is None:
        raise ImportError("PyYAML is required when using --config. Please install it via `pip install pyyaml`.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def extract_mel_f0_from_wav(
    wav_path: str,
    sampling_rate: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    num_mels: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从单个 wav 中提取 mel 频谱与 F0（使用 torchfcpe），返回:
      mel: (1, num_mels, frames)
      f0:  (1, frames)
    """
    # 读入并裁剪到 [-1, 1]
    audio = load_wav(wav_path, sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    # 与训练/预处理保持一致，使用略保守的裁剪范围
    audio = np.clip(audio, -0.95, 0.95)

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

    # 使用统一工具提取 F0，保证与训练管线对齐
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
    return mel, f0_t


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    sampling_rate = data_cfg["sampling_rate"]
    n_fft = data_cfg["n_fft"]
    hop_size = data_cfg["hop_size"]
    win_size = data_cfg["win_size"]
    fmin = data_cfg["fmin"]
    fmax = data_cfg["fmax"]
    num_mels = data_cfg["num_mels"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建模型并加载权重（保持与 train_nsf.py 一致）
    model = NsfHifiGanModel(
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        lr=model_cfg.get("lr", 2e-4),
        beta1=model_cfg.get("beta1", 0.8),
        beta2=model_cfg.get("beta2", 0.99),
        upsample_initial_channel=model_cfg.get("upsample_initial_channel", 512),
        upsample_rates=model_cfg.get("upsample_rates", [8, 8, 2, 2, 2]),
        upsample_kernel_sizes=model_cfg.get("upsample_kernel_sizes", [16, 16, 4, 4, 4]),
        resblock=model_cfg.get("resblock", "1"),
        resblock_kernel_sizes=model_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilation_sizes=model_cfg.get(
            "resblock_dilation_sizes",
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        ),
        discriminator_periods=model_cfg.get(
            "discriminator_periods", [2, 3, 5, 7, 11]
        ),
        mini_nsf=model_cfg.get("mini_nsf", False),
        noise_sigma=model_cfg.get("noise_sigma", 0.0),
        loss_fft_sizes=tuple(model_cfg.get("loss_fft_sizes", [512, 1024, 2048])),
        loss_hop_sizes=tuple(model_cfg.get("loss_hop_sizes", [128, 256, 512])),
        loss_win_lengths=tuple(model_cfg.get("loss_win_lengths", [512, 1024, 2048])),
        aux_mel_weight=model_cfg.get("aux_mel_weight", 45.0),
        aux_stft_weight=model_cfg.get("aux_stft_weight", 2.5),
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # 提取 mel + F0
    mel, f0 = extract_mel_f0_from_wav(
        args.wav,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
        num_mels=num_mels,
        device=device,
    )

    with torch.no_grad():
        wav_out = model.generator(mel.to(device), f0.to(device))  # (1, 1, T)
    wav_out = wav_out.squeeze().cpu().numpy()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(args.out, wav_out, samplerate=sampling_rate)
    print(f"Saved NSF vocoder output to: {args.out}")


if __name__ == "__main__":
    main()
