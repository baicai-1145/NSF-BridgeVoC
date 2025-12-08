from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from div.data_module import mel_spectrogram
from div.nsf import (
    AttrDict,
    NsfHifiGenerator,
    NsfMultiPeriodDiscriminator,
    NsfMultiScaleDiscriminator,
)
from div.nsf.loss import NsfAuxLoss, nsf_d_loss, nsf_g_loss


class NsfHifiGanModel(pl.LightningModule):
    """
    NSF-HiFiGAN 声码器的 Lightning 封装（阶段 1+2）。

    - 生成器结构直接来自 SingingVocoders 的 nsf_HiFigan。
    - 损失包含：判别器 LSGAN + feature matching + Mel + 多分辨率 STFT。
    """

    def __init__(
        self,
        # 模型结构
        sampling_rate: int = 44100,
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        upsample_rates: List[int] = None,
        upsample_kernel_sizes: List[int] = None,
        resblock: str = "1",
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
        discriminator_periods: List[int] = None,
        mini_nsf: bool = False,
        noise_sigma: float = 0.0,
        # 训练与损失
        lr: float = 2e-4,
        beta1: float = 0.8,
        beta2: float = 0.99,
        n_fft: int = 2048,
        hop_size: int = 512,
        win_size: int = 2048,
        fmin: float = 0.0,
        fmax: float = 22050.0,
        loss_fft_sizes=(512, 1024, 2048),
        loss_hop_sizes=(128, 256, 512),
        loss_win_lengths=(512, 1024, 2048),
        aux_mel_weight: float = 45.0,
        aux_stft_weight: float = 2.5,
    ):
        super().__init__()
        if upsample_rates is None:
            # 与 SingingVocoders nsf_hifigan 配置保持一致，乘积=hop_size=512
            upsample_rates = [8, 8, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ]
        if discriminator_periods is None:
            discriminator_periods = [2, 3, 5, 7, 11]

        h_dict = dict(
            sampling_rate=sampling_rate,
            num_mels=num_mels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            discriminator_periods=discriminator_periods,
            mini_nsf=mini_nsf,
            noise_sigma=noise_sigma,
        )
        h = AttrDict(h_dict)

        self.generator = NsfHifiGenerator(h)
        self.discriminator = nn.ModuleDict(
            {
                "msd": NsfMultiScaleDiscriminator(),
                "mpd": NsfMultiPeriodDiscriminator(periods=discriminator_periods),
            }
        )

        self.aux_loss = NsfAuxLoss(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            fmin=fmin,
            fmax=fmax,
            n_mels=num_mels,
            fft_sizes=loss_fft_sizes,
            hop_sizes=loss_hop_sizes,
            win_lengths=loss_win_lengths,
            mel_weight=aux_mel_weight,
            stft_weight=aux_stft_weight,
            use_stft=True,
        )

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.save_hyperparameters()

        # 手动优化 G/D
        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        opt_d = torch.optim.AdamW(
            list(self.discriminator["msd"].parameters())
            + list(self.discriminator["mpd"].parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        return [opt_g, opt_d]

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        推理接口：mel (B, n_mels, frames), f0 (B, frames)
        返回 (B, 1, T)
        """
        return self.generator(mel, f0)

    def _d_forward(self, audio: torch.Tensor):
        msd_out, msd_feat = self.discriminator["msd"](audio)
        mpd_out, mpd_feat = self.discriminator["mpd"](audio)
        return (msd_out, msd_feat), (mpd_out, mpd_feat)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        opt_g, opt_d = self.optimizers()

        audio = batch["audio"]  # (B, 1, T)
        mel = batch["mel"]  # (B, n_mels, frames)
        f0 = batch["f0"]  # (B, frames)

        # 1) 生成器前向
        wav_fake = self.generator(mel, f0)

        # 2) 更新判别器
        for p in self.discriminator.parameters():
            p.requires_grad = True

        Dfake = self._d_forward(wav_fake.detach())
        Dtrue = self._d_forward(audio)
        Dloss, Dlog = nsf_d_loss(Dfake, Dtrue)

        opt_d.zero_grad()
        self.manual_backward(Dloss)
        opt_d.step()

        # 3) 更新生成器
        for p in self.discriminator.parameters():
            p.requires_grad = False

        GDfake = self._d_forward(wav_fake)
        GDtrue = self._d_forward(audio)
        G_adv_fm, Glog = nsf_g_loss(GDfake, GDtrue)
        aux_loss, Auxlog = self.aux_loss(wav_fake, audio)
        Gloss = G_adv_fm + aux_loss

        opt_g.zero_grad()
        self.manual_backward(Gloss)
        opt_g.step()

        log_dict = {
            "loss_D": Dloss.detach(),
            "loss_G": Gloss.detach(),
        }
        for k, v in {**Dlog, **Glog, **Auxlog}.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] = v.detach()
        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": Gloss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        audio = batch["audio"]
        mel = batch["mel"]
        f0 = batch["f0"]
        with torch.no_grad():
            wav_fake = self.generator(mel, f0)
            aux_loss, _ = self.aux_loss(wav_fake, audio)
        self.log("val_aux_loss", aux_loss, on_step=False, on_epoch=True, prog_bar=True)

        # 首个 batch 额外在 TensorBoard 中记录 mel 频谱与 F0，方便检查 F0 管线与重建质量
        if batch_idx == 0 and hasattr(self.logger, "experiment"):
            try:
                tb = self.logger.experiment
                import matplotlib.pyplot as plt

                sr = self.hparams.sampling_rate
                n_fft = self.hparams.n_fft
                hop_size = self.hparams.hop_size
                win_size = self.hparams.win_size
                fmin = self.hparams.fmin
                fmax = self.hparams.fmax
                num_mels = self.hparams.num_mels

                # 由波形重新计算 mel，和数据集提供的 mel 对比
                mel_pred = mel_spectrogram(
                    wav_fake[:, 0],  # (B, T)
                    n_fft=n_fft,
                    num_mels=num_mels,
                    sampling_rate=sr,
                    hop_size=hop_size,
                    win_size=win_size,
                    fmin=fmin,
                    fmax=fmax,
                    center=True,
                    in_dataset=False,
                )
                mel_gt = mel
                # [diff | gt | pred]
                spec_cat = torch.cat([(mel_pred - mel_gt).abs(), mel_gt, mel_pred], dim=2)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.pcolor(spec_cat[0].cpu().numpy())
                ax.set_title("NSF mel diff | GT | Pred")
                ax.set_xlabel("Frames")
                ax.set_ylabel("Mel bins")
                plt.tight_layout()
                tb.add_figure("validation/nsf_mel", fig, global_step=self.global_step)
                plt.close(fig)

                # F0 曲线可视化，检查是否存在明显错误（全 0、跳变等）
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(f0[0].detach().cpu().numpy())
                ax2.set_title("NSF F0 (Hz)")
                ax2.set_xlabel("Frames")
                ax2.set_ylabel("F0")
                plt.tight_layout()
                tb.add_figure("validation/nsf_f0", fig2, global_step=self.global_step)
                plt.close(fig2)

                # 同时记录一对音频
                tb.add_audio(
                    "validation/nsf_audio_pred",
                    wav_fake[0],
                    sample_rate=sr,
                    global_step=self.global_step,
                )
                tb.add_audio(
                    "validation/nsf_audio_gt",
                    audio[0],
                    sample_rate=sr,
                    global_step=self.global_step,
                )
            except Exception as e:
                print(f"[WARN] NSF validation visualization failed: {e}")

        return {"val_aux_loss": aux_loss}
