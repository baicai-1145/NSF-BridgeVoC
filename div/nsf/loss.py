from typing import Dict, Tuple

import torch
import torch.nn as nn

from div.util.loss import (
    MultiresolutionSTFTLoss,
    dynamic_range_compression_torch,
    mel_spectrogram,
)
from . import (
    nsf_discriminator_loss,
    nsf_feature_loss,
    nsf_generator_loss,
)


class NsfAuxLoss(nn.Module):
    """
    NSF 辅助重建损失：Mel + 多分辨率 STFT。

    设计思路参考 SingingVocoders 中的 HiFiloss / univloss，
    但实现上复用本项目已有的 mel_spectrogram 和 MultiresolutionSTFTLoss。
    """

    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        n_mels: int,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(128, 256, 512),
        win_lengths=(512, 1024, 2048),
        mel_weight: float = 45.0,
        stft_weight: float = 2.5,
        use_stft: bool = True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.mel_weight = mel_weight
        self.stft_weight = stft_weight
        self.use_stft = use_stft
        if use_stft:
            self.mrstft = MultiresolutionSTFTLoss(
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=win_lengths,
            )

    def forward(
        self, audio_pred: torch.Tensor, audio_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        audio_pred, audio_gt: (B, 1, T) 或 (B, T)
        """
        if audio_pred.dim() == 3:
            audio_pred = audio_pred.squeeze(1)
        if audio_gt.dim() == 3:
            audio_gt = audio_gt.squeeze(1)

        # Mel loss
        mel_pred = mel_spectrogram(
            audio_pred,
            n_fft=self.n_fft,
            num_mels=self.n_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        mel_gt = mel_spectrogram(
            audio_gt,
            n_fft=self.n_fft,
            num_mels=self.n_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        mel_pred = dynamic_range_compression_torch(mel_pred)
        mel_gt = dynamic_range_compression_torch(mel_gt)
        mel_loss = torch.mean(torch.abs(mel_pred - mel_gt)) * self.mel_weight

        total_loss = mel_loss
        logs: Dict[str, torch.Tensor] = {"aux_mel_loss": mel_loss}

        if self.use_stft:
            stft_loss = self.mrstft(audio_gt, audio_pred, None) * self.stft_weight
            total_loss = total_loss + stft_loss
            logs["aux_stft_loss"] = stft_loss

        return total_loss, logs


def nsf_d_loss(Dfake, Dtrue) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    判别器损失，兼容 (msd, mpd) 或 (mrd, mpd) 的输出结构。
    Dfake / Dtrue: ((disc1_out, _), (disc2_out, _))
    """
    (Fdisc1_out, _), (Fdisc2_out, _) = Dfake
    (Tdisc1_out, _), (Tdisc2_out, _) = Dtrue

    loss1, d1_r, d1_g = nsf_discriminator_loss(Tdisc1_out, Fdisc1_out)
    loss2, d2_r, d2_g = nsf_discriminator_loss(Tdisc2_out, Fdisc2_out)
    loss = loss1 + loss2

    log = {
        "Ddisc1_real": float(sum(d1_r) / max(len(d1_r), 1)),
        "Ddisc1_fake": float(sum(d1_g) / max(len(d1_g), 1)),
        "Ddisc2_real": float(sum(d2_r) / max(len(d2_r), 1)),
        "Ddisc2_fake": float(sum(d2_g) / max(len(d2_g), 1)),
    }
    return loss, log


def nsf_g_loss(GDfake, GDtrue) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    生成器对抗 + feature matching 损失。
    GDfake / GDtrue: ((disc1_out, disc1_fmap), (disc2_out, disc2_fmap))
    """
    (disc1_out_f, disc1_fmap_f), (disc2_out_f, disc2_fmap_f) = GDfake
    (_, disc1_fmap_t), (_, disc2_fmap_t) = GDtrue

    g_loss1, _ = nsf_generator_loss(disc1_out_f)
    g_loss2, _ = nsf_generator_loss(disc2_out_f)
    fm_loss1 = nsf_feature_loss(disc1_fmap_t, disc1_fmap_f)
    fm_loss2 = nsf_feature_loss(disc2_fmap_t, disc2_fmap_f)

    g_loss = g_loss1 + g_loss2
    fm_loss = fm_loss1 + fm_loss2

    total = g_loss + fm_loss
    log = {
        "G_adv_loss": g_loss,
        "G_fm_loss": fm_loss,
    }
    return total, log

