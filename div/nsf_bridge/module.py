from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from div.backbones.bcd import BCD
from div.data_module import inverse_mel, mel_spectrogram
from div.nsf import AttrDict, NsfMultiPeriodDiscriminator, NsfMultiScaleDiscriminator
from div.nsf.loss import NsfAuxLoss, nsf_d_loss, nsf_g_loss
from div.nsf.nsf_hifigan import SourceModuleHnNSF


class NsfBridgeGenerator(nn.Module):
    """
    NSF 源 + BCD 子带解码器：

    - 输入: mel (B, n_mels, frames), f0 (B, frames)
    - 步骤:
      1) 用 NSF 源根据 f0 生成谐波激励波形；
      2) 从 mel 通过 inverse_mel 近似恢复幅度谱；
      3) 将 NSF 激励的相位与 mel 幅度结合，形成条件谱；
      4) 用 BCD 在 STFT 域上做映射，输出目标复谱；
      5) iSTFT 得到波形。
    """

    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        num_mels: int,
        # BCD 相关超参
        nblocks: int,
        hidden_channel: int,
        f_kernel_size: int,
        t_kernel_size: int,
        mlp_ratio: int,
        ada_rank: int,
        ada_alpha: int,
        ada_mode: str,
        act_type: str,
        pe_type: str,
        scale: int,
        decode_type: str,
        use_adanorm: bool,
        causal: bool,
        # NSF 源相关
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels

        # NSF 源：根据 F0 生成谐波激励
        self.source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            add_noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
        )

        # BCD 子带网络作为 STFT 域解码器
        self.bcd = BCD(
            nblocks=nblocks,
            hidden_channel=hidden_channel,
            f_kernel_size=f_kernel_size,
            t_kernel_size=t_kernel_size,
            mlp_ratio=mlp_ratio,
            ada_rank=ada_rank,
            ada_alpha=ada_alpha,
            ada_mode=ada_mode,
            input_channel=4,  # cat(inpt(2), cond(2))
            act_type=act_type,
            pe_type=pe_type,
            scale=scale,
            decode_type=decode_type,
            use_adanorm=use_adanorm,
            causal=causal,
            sampling_rate=sampling_rate,
        )

        self.register_buffer(
            "stft_window", torch.hann_window(self.win_size), persistent=False
        )

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T) or (B, T)
        返回: complex STFT (B, F, T_frames)
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return torch.stft(
            wav,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.stft_window.to(wav.device),
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """
        spec: complex (B, F, T_frames)
        返回: (B, 1, length)
        """
        wav = torch.istft(
            spec,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.stft_window.to(spec.device),
            center=True,
            length=length,
        )
        return wav.unsqueeze(1)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, n_mels, frames)
        f0:  (B, frames)
        返回: (B, 1, T)
        """
        B, _, frames = mel.shape
        device = mel.device

        # 1) NSF 源生成谐波激励（使用 hop_size 作为上采样因子）
        har_source = self.source(f0, self.hop_size)  # (B, T, 1)
        har_source = har_source.transpose(1, 2)  # (B, 1, T)

        # 2) STFT 得到激励的相位
        spec_src = self._stft(har_source)  # (B, F, T_src)
        pha_src = torch.angle(spec_src)

        # 3) 由 mel 近似恢复幅度谱
        # mel 本身已经是 log-mel，inverse_mel 内部会做 exp 反变换
        mag_mel = inverse_mel(
            mel,
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
            in_dataset=False,
        ).abs().clamp_min_(1e-6)  # (B, F, T_mel)

        # 对齐时间维度（通常 T_src ≈ T_mel）
        T_common = min(spec_src.shape[-1], mag_mel.shape[-1])
        spec_src = spec_src[..., :T_common]
        pha_src = pha_src[..., :T_common]
        mag_mel = mag_mel[..., :T_common]

        # 4) 组合 NSF 相位 + mel 幅度，作为条件谱
        cond_spec = torch.complex(
            mag_mel * torch.cos(pha_src), mag_mel * torch.sin(pha_src)
        )  # (B, F, T)
        cond_ri = torch.stack([cond_spec.real, cond_spec.imag], dim=1)  # (B, 2, F, T)

        # 输入谱可以先设为 0，完全依赖 cond + BCD 学习映射
        inpt = torch.zeros_like(cond_ri)
        t = torch.zeros(B, device=device)

        out_ri = self.bcd(inpt, cond_ri, time_cond=t)  # (B, 2, F, T)
        out_spec = torch.complex(out_ri[:, 0], out_ri[:, 1])

        # 5) iSTFT 还原波形，长度与激励一致
        target_len = frames * self.hop_size
        wav = self._istft(out_spec, length=target_len)
        return wav


class NsfBridgeVocModel(pl.LightningModule):
    """
    真正的 NSF-BridgeVoc 模型：

    - 生成器: NsfBridgeGenerator (NSF 源 + BCD 子带解码器)
    - 判别器与损失: 复用 nsf-HiFiGAN 的 MultiScale/MultiPeriod + Mel/STFT 辅助损失
    """

    def __init__(
        self,
        sampling_rate: int = 44100,
        num_mels: int = 128,
        # 频谱参数
        n_fft: int = 2048,
        hop_size: int = 512,
        win_size: int = 2048,
        fmin: float = 0.0,
        fmax: float = 22050.0,
        # 优化与损失
        lr: float = 2e-4,
        beta1: float = 0.8,
        beta2: float = 0.99,
        loss_fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        loss_hop_sizes: Tuple[int, ...] = (128, 256, 512),
        loss_win_lengths: Tuple[int, ...] = (512, 1024, 2048),
        aux_mel_weight: float = 45.0,
        aux_stft_weight: float = 2.5,
        # BCD 结构参数（与默认 BridgeVoC 保持一致）
        nblocks: int = 8,
        hidden_channel: int = 256,
        f_kernel_size: int = 9,
        t_kernel_size: int = 11,
        mlp_ratio: int = 1,
        ada_rank: int = 32,
        ada_alpha: int = 32,
        ada_mode: str = "sola",
        act_type: str = "gelu",
        pe_type: str = "positional",
        scale: int = 1000,
        decode_type: str = "ri",
        use_adanorm: bool = True,
        causal: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        # 生成器：NSF 源 + BCD 解码器
        self.generator = NsfBridgeGenerator(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            fmin=fmin,
            fmax=fmax,
            num_mels=num_mels,
            nblocks=nblocks,
            hidden_channel=hidden_channel,
            f_kernel_size=f_kernel_size,
            t_kernel_size=t_kernel_size,
            mlp_ratio=mlp_ratio,
            ada_rank=ada_rank,
            ada_alpha=ada_alpha,
            ada_mode=ada_mode,
            act_type=act_type,
            pe_type=pe_type,
            scale=scale,
            decode_type=decode_type,
            use_adanorm=use_adanorm,
            causal=causal,
        )

        # 判别器：沿用 nsf-HiFiGAN 的 msd/mpd
        self.discriminator = nn.ModuleDict(
            {
                "msd": NsfMultiScaleDiscriminator(),
                "mpd": NsfMultiPeriodDiscriminator(),
            }
        )

        # 辅助损失：mel L1 + 多分辨率 STFT
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
        推理接口：mel (B, n_mels, frames), f0 (B, frames) -> (B, 1, T)
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

                mel_pred = mel_spectrogram(
                    wav_fake[:, 0],
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
                spec_cat = torch.cat([(mel_pred - mel_gt).abs(), mel_gt, mel_pred], dim=2)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.pcolor(spec_cat[0].cpu().numpy())
                ax.set_title("NSF-Bridge mel diff | GT | Pred")
                ax.set_xlabel("Frames")
                ax.set_ylabel("Mel bins")
                plt.tight_layout()
                tb.add_figure("validation/nsf_bridge_mel", fig, global_step=self.global_step)
                plt.close(fig)

                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(f0[0].detach().cpu().numpy())
                ax2.set_title("NSF-Bridge F0 (Hz)")
                ax2.set_xlabel("Frames")
                ax2.set_ylabel("F0")
                plt.tight_layout()
                tb.add_figure("validation/nsf_bridge_f0", fig2, global_step=self.global_step)
                plt.close(fig2)

                tb.add_audio(
                    "validation/nsf_bridge_audio_pred",
                    wav_fake[0],
                    sample_rate=sr,
                    global_step=self.global_step,
                )
                tb.add_audio(
                    "validation/nsf_bridge_audio_gt",
                    audio[0],
                    sample_rate=sr,
                    global_step=self.global_step,
                )
            except Exception as e:
                print(f"[WARN] NSF-Bridge validation visualization failed: {e}")

        return {"val_aux_loss": aux_loss}

