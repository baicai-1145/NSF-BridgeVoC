from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Mel2MagConfig:
    num_mels: int
    n_freq: int
    hidden: int = 256
    n_blocks: int = 6
    kernel_size: int = 5
    dropout: float = 0.0
    f0_max: float = 1100.0
    eps: float = 1e-6


class _Res1DBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(F.gelu(self.norm2(x)))
        return x + residual


class Mel2MagHF(nn.Module):
    """
    从 log-mel(+f0/uv) 预测线性 STFT 幅度 (mag_hat)。

    输入:
      mel: (B, n_mels, T)  # 来自 div.data_module.mel_spectrogram 的 log-mel
      f0:  (B, T)
      uv:  (B, T) or None
    输出:
      mag_hat: (B, n_freq, T)  # 线性幅度，>=0
    """

    def __init__(self, cfg: Mel2MagConfig):
        super().__init__()
        if cfg.num_mels <= 0:
            raise ValueError(f"num_mels must be positive, got {cfg.num_mels}")
        if cfg.n_freq <= 0:
            raise ValueError(f"n_freq must be positive, got {cfg.n_freq}")
        if cfg.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {cfg.hidden}")
        if cfg.n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {cfg.n_blocks}")
        if cfg.kernel_size <= 0 or cfg.kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be positive odd, got {cfg.kernel_size}")

        self.cfg = cfg
        in_ch = int(cfg.num_mels) + 2  # mel + f0 + uv
        self.in_proj = nn.Conv1d(in_ch, int(cfg.hidden), kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _Res1DBlock(int(cfg.hidden), kernel_size=int(cfg.kernel_size), dropout=float(cfg.dropout))
                for _ in range(int(cfg.n_blocks))
            ]
        )
        self.out_norm = nn.GroupNorm(1, int(cfg.hidden))
        self.out_proj = nn.Conv1d(int(cfg.hidden), int(cfg.n_freq), kernel_size=1)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, uv: torch.Tensor | None = None) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"mel must be (B,n_mels,T), got shape={tuple(mel.shape)}")
        if f0.ndim != 2:
            raise ValueError(f"f0 must be (B,T), got shape={tuple(f0.shape)}")
        B, n_mels, T = mel.shape
        if n_mels != int(self.cfg.num_mels):
            raise ValueError(f"Expected mel n_mels={self.cfg.num_mels}, got {n_mels}")
        if f0.shape[0] != B:
            raise ValueError(f"f0 batch mismatch: mel B={B}, f0 B={f0.shape[0]}")

        if uv is None:
            uv = (f0 > 0).to(mel.dtype)
        if uv.ndim != 2:
            raise ValueError(f"uv must be (B,T), got shape={tuple(uv.shape)}")

        T_common = min(int(T), int(f0.shape[-1]), int(uv.shape[-1]))
        if T_common <= 0:
            raise ValueError("Empty time dimension after alignment")
        mel = mel[..., :T_common]
        f0 = f0[..., :T_common]
        uv = uv[..., :T_common]

        f0 = torch.clamp(f0, min=0.0)
        f0_norm = torch.log1p(f0) / torch.log1p(torch.tensor(float(self.cfg.f0_max), device=f0.device, dtype=f0.dtype))
        f0_norm = torch.clamp(f0_norm, 0.0, 1.0)

        x = torch.cat([mel, f0_norm.unsqueeze(1), uv.unsqueeze(1).to(mel.dtype)], dim=1)  # (B, n_mels+2, T)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(F.gelu(self.out_norm(x)))
        mag = F.softplus(x) + float(self.cfg.eps)
        return mag

