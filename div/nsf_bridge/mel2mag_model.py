from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mel2mag import Mel2MagConfig, Mel2MagHF


def _hz_to_bin(hz: float, *, sr: int, n_fft: int) -> int:
    hz_per_bin = float(sr) / float(n_fft)
    return int(round(float(hz) / hz_per_bin))


class Mel2MagLightning(pl.LightningModule):
    """
    预训练：mel(+f0/uv) -> 线性幅度 |STFT(x)|

    - 只训练幅度回归，不引入 GAN
    - 通过高频门控（按 HF 能量占比）降低静音段 hiss 风险
    """

    def __init__(
        self,
        *,
        sampling_rate: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        num_mels: int,
        drop_last_freq: bool = True,
        # model
        hidden: int = 256,
        n_blocks: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.0,
        f0_max: float = 1100.0,
        # optim
        lr: float = 2e-4,
        beta1: float = 0.8,
        beta2: float = 0.99,
        opt_type: str = "AdamW",
        # loss
        hf_fmin_hz: float = 6000.0,
        hf_fmax_hz: float = 15000.0,
        hf_weight: float = 1.0,
        hf_gate_ratio: float = 0.01,
        edge_weight: float = 0.0,
        eps: float = 1e-6,
        **unused_kwargs: Any,
    ):
        super().__init__()

        self.sampling_rate = int(sampling_rate)
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.win_size = int(win_size)
        self.num_mels = int(num_mels)
        self.drop_last_freq = bool(drop_last_freq)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.opt_type = str(opt_type)
        self.hf_fmin_hz = float(hf_fmin_hz)
        self.hf_fmax_hz = float(hf_fmax_hz)
        self.hf_weight = float(hf_weight)
        self.hf_gate_ratio = float(hf_gate_ratio)
        self.edge_weight = float(edge_weight)
        self.eps = float(eps)

        n_freq = (self.n_fft // 2 + 1) - (1 if self.drop_last_freq else 0)
        cfg = Mel2MagConfig(
            num_mels=self.num_mels,
            n_freq=n_freq,
            hidden=int(hidden),
            n_blocks=int(n_blocks),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            f0_max=float(f0_max),
            eps=float(eps),
        )
        self.mel2mag = Mel2MagHF(cfg)
        self.register_buffer("stft_window", torch.hann_window(self.win_size), persistent=False)

        # 保存超参便于 ckpt/恢复
        self.save_hyperparameters(ignore=["unused_kwargs"])

    def configure_optimizers(self):
        params = self.parameters()
        if self.opt_type.lower() == "adam":
            return torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))
        return torch.optim.AdamW(params, lr=self.lr, betas=(self.beta1, self.beta2))

    def _stft_mag(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, 1, T) or (B, T)
        return: (B, F, T_frames)
        """
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.stft_window.to(audio.device),
            center=True,
            return_complex=True,
        )
        mag = spec.abs().clamp_min(self.eps)
        if self.drop_last_freq:
            mag = mag[:, :-1, :].contiguous()
        return mag

    def _compute_losses(
        self, *, mag_hat: torch.Tensor, mag_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        mag_hat/mag_gt: (B, F, T)
        """
        B = int(mag_gt.shape[0])
        if B <= 0:
            raise ValueError("Empty batch")

        T_common = min(int(mag_hat.shape[-1]), int(mag_gt.shape[-1]))
        F_common = min(int(mag_hat.shape[-2]), int(mag_gt.shape[-2]))
        mag_hat = mag_hat[:, :F_common, :T_common]
        mag_gt = mag_gt[:, :F_common, :T_common]

        loss_mag = torch.mean(torch.abs(mag_hat - mag_gt))

        # 高频门控：仅当 HF 能量占比足够高时才施加 HF loss，避免静音段学出 hiss
        b0 = max(0, _hz_to_bin(self.hf_fmin_hz, sr=self.sampling_rate, n_fft=self.n_fft))
        b1 = min(F_common - 1, _hz_to_bin(self.hf_fmax_hz, sr=self.sampling_rate, n_fft=self.n_fft))
        if b1 <= b0:
            hf_loss = mag_hat.new_zeros(())
            hf_gate_mean = mag_hat.new_zeros(())
        else:
            hf_gt = mag_gt[:, b0 : b1 + 1]
            hf_pr = mag_hat[:, b0 : b1 + 1]
            # per-sample energy ratio
            e_all = mag_gt.mean(dim=(1, 2)).clamp_min(self.eps)
            e_hf = hf_gt.mean(dim=(1, 2))
            ratio = e_hf / e_all
            gate = (ratio > self.hf_gate_ratio).to(mag_gt.dtype)  # (B,)
            hf_l1_per = torch.mean(torch.abs(hf_pr - hf_gt), dim=(1, 2))
            hf_loss = torch.sum(hf_l1_per * gate) / torch.clamp(gate.sum(), min=1.0)
            hf_gate_mean = gate.mean()

        # 频率梯度对齐（可选）：减少“涂抹”
        if self.edge_weight > 0:
            log_gt = torch.log(mag_gt + self.eps)
            log_pr = torch.log(mag_hat + self.eps)
            grad_gt = log_gt[:, 1:, :] - log_gt[:, :-1, :]
            grad_pr = log_pr[:, 1:, :] - log_pr[:, :-1, :]
            loss_edge = torch.mean(torch.abs(grad_pr - grad_gt))
        else:
            loss_edge = mag_hat.new_zeros(())

        loss = loss_mag + self.hf_weight * hf_loss + self.edge_weight * loss_edge
        logs = {
            "loss_mag": loss_mag.detach(),
            "loss_hf": hf_loss.detach(),
            "hf_gate_mean": hf_gate_mean.detach(),
            "loss_edge": loss_edge.detach(),
        }
        return loss, logs

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        audio = batch["audio"]  # (B,1,T)
        f0 = batch["f0"]  # (B,T_frames)
        mel = batch["mel"]  # (B,n_mels,T_frames)
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        if f0.ndim == 1:
            f0 = f0.unsqueeze(0)
        uv = (f0 > 0).to(mel.dtype)
        mag_hat = self.mel2mag(mel, f0, uv)
        mag_gt = self._stft_mag(audio)

        # 对齐到同一时间长度
        T_common = min(int(mag_hat.shape[-1]), int(mag_gt.shape[-1]), int(mel.shape[-1]), int(f0.shape[-1]))
        mag_hat = mag_hat[..., :T_common]
        mag_gt = mag_gt[..., :T_common]
        return mag_hat, mag_gt

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        mag_hat, mag_gt = self._forward_batch(batch)
        loss, logs = self._compute_losses(mag_hat=mag_hat, mag_gt=mag_gt)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        mag_hat, mag_gt = self._forward_batch(batch)
        loss, logs = self._compute_losses(mag_hat=mag_hat, mag_gt=mag_gt)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        return {"loss": loss.detach(), **logs}

