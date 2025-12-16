from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

from div.backbones.bcd_nsf_bridge import NsfBcdBridge
from div.backbones.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from div.sdes import BridgeGAN
from div.util.loss import (
    MultiresolutionMelLoss,
    MultiresolutionSTFTLoss,
    MelLoss,
    FeatureMatchingLoss,
    DiscriminatorLoss,
    GeneratorLoss,
)
from div.data_module import spectral_normalize_torch


class NsfBridgeScoreModel(pl.LightningModule):
    """
    NSF-BridgeVoc 的 Score-based 训练模型：

    - 生成器/backbone: NsfBcdBridge (NSF 源 + BCD 子带 Bridge)；
    - SDE: BridgeGAN（与 bridge-only 一致的 score-based bridge 过程）；
    - 损失: score_mse + multi-mel / multi-stft + 可选 GAN（与 ScoreModelGAN 相同风格）。

    输入 batch 结构来自 NsfBridgeDataset：
      {
        "audio": (B, 1, T),
        "mel":   (B, num_mels, frames),
        "f0":    (B, frames),
      }
    """

    def __init__(
        self,
        # 数据 / 频谱参数
        sampling_rate: int = 44100,
        n_fft: int = 2048,
        hop_size: int = 512,
        win_size: int = 2048,
        fmin: float = 0.0,
        fmax: float = 22050.0,
        num_mels: int = 128,
        # 频谱压缩参数（与 bridge-only DataModule 对齐）
        spec_factor: float = 0.33,
        spec_abs_exponent: float = 0.5,
        transform_type: str = "exponent",
        drop_last_freq: bool = True,
        # ScoreModel & 优化参数（与 ScoreModelGAN 对齐）
        opt_type: str = "AdamW",
        lr: float = 5e-4,
        beta1: float = 0.8,
        beta2: float = 0.99,
        ema_decay: float = 0.999,
        t_eps: float = 0.03,
        loss_type_list: str = "score_mse:1.0,multi-mel:0.4,multi-stft:0.2",
        use_gan: bool = True,
        num_eval_files: int = 20,
        max_epochs: int = 1000,
        lr_scheduler_interval: str = "epoch",
        lr_eta_min: float = 1e-5,
        lr_tmax_steps: int = 0,
        # BridgeGAN SDE 参数（与 default_bridgevoc_44k1.yaml 对齐）
        beta_min: float = 0.01,
        beta_max: float = 20.0,
        c: float = 0.4,
        k: float = 2.6,
        bridge_type: str = "gmax",
        N: int = 4,
        offset: float = 1e-5,
        predictor: str = "x0",
        sampling_type: str = "sde_first_order",
        # BCD 结构参数（与 bridge-only 保持一致）
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
        # NSF 源参数
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        phase_mask_ratio: float = 0.1,
        mel_phase_gate_ratio: float = 0.0,
        # BCD high-SR band mode (forwarded into BCD)
        highsr_band_mode: str = "legacy",
        highsr_freq_bins: int = 1024,
        highsr_coarse_stride_f: int = 16,
        highsr_refine8_start: int = 256,
        highsr_refine4_start: int = 672,
        highsr_refine_overlap: int = 64,
        highsr_refine8_nblocks: int = 4,
        highsr_refine4_nblocks: int = 2,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.drop_last_freq = drop_last_freq
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.transform_type = transform_type

        self.register_buffer(
            "stft_window", torch.hann_window(self.win_size, periodic=True), persistent=False
        )

        try:
            from librosa.filters import mel as librosa_mel_fn
        except Exception as e:
            raise ImportError("librosa is required for NsfBridgeScoreModel mel basis.") from e

        mel = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float(), persistent=False)

        # SDE：BridgeGAN
        self.sde = BridgeGAN(
            beta_min=beta_min,
            beta_max=beta_max,
            c=c,
            k=k,
            bridge_type=bridge_type,
            N=N,
            offset=offset,
            predictor=predictor,
            sampling_type=sampling_type,
        )
        self.sde_name = "bridgegan"
        self.t_eps = t_eps

        # 生成器 / backbone：NSF 源 + BCD
        self.dnn = NsfBcdBridge(
            nblocks=nblocks,
            hidden_channel=hidden_channel,
            f_kernel_size=f_kernel_size,
            t_kernel_size=t_kernel_size,
            mlp_ratio=mlp_ratio,
            ada_rank=ada_rank,
            ada_alpha=ada_alpha,
            ada_mode=ada_mode,
            input_channel=4,
            act_type=act_type,
            pe_type=pe_type,
            scale=scale,
            decode_type=decode_type,
            use_adanorm=use_adanorm,
            causal=causal,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_size=hop_size,
            win_size=win_size,
            fmin=fmin,
            fmax=fmax,
            num_mels=num_mels,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            add_noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
            phase_mask_ratio=phase_mask_ratio,
            mel_phase_gate_ratio=mel_phase_gate_ratio,
            highsr_band_mode=highsr_band_mode,
            highsr_freq_bins=highsr_freq_bins,
            highsr_coarse_stride_f=highsr_coarse_stride_f,
            highsr_refine8_start=highsr_refine8_start,
            highsr_refine4_start=highsr_refine4_start,
            highsr_refine_overlap=highsr_refine_overlap,
            highsr_refine8_nblocks=highsr_refine8_nblocks,
            highsr_refine4_nblocks=highsr_refine4_nblocks,
        )

        # GAN 判别器（与 ScoreModelGAN 相同）
        self.opt_type = opt_type
        self.use_gan = use_gan
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_epochs = max_epochs
        self.lr_scheduler_interval = str(lr_scheduler_interval).lower()
        self.lr_eta_min = float(lr_eta_min)
        self.lr_tmax_steps = int(lr_tmax_steps)
        self.num_eval_files = num_eval_files

        if self.use_gan:
            self.mpd = MultiPeriodDiscriminator()
            self.mrd = MultiResolutionDiscriminator()
            self.optim_d = torch.optim.AdamW(
                list(self.mpd.parameters()) + list(self.mrd.parameters()),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )
            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                self.optim_d, gamma=0.999, last_epoch=-1
            )

        # EMA（仅对生成器参数）
        from torch_ema import ExponentialMovingAverage

        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        # loss 配置
        if isinstance(loss_type_list, str):
            loss_type_list = loss_type_list.strip().split(",")
        self.loss_type_list = loss_type_list
        self._reduce_op_3 = (
            lambda x: torch.mean(torch.sum(x, dim=[1, 2])) if x.ndim == 3 else torch.mean(torch.sum(x, dim=[1, 2, 3]))
        )
        self._reduce_op_4 = (
            lambda x: torch.mean(torch.sum(x, dim=[1, 2, 3])) if x.ndim == 4 else torch.mean(torch.sum(x, dim=[1, 2]))
        )

        self.weight_dict: Dict[str, float] = {}
        self.loss_dict: Dict[str, nn.Module] = {}
        for cur_loss_zip in loss_type_list:
            cur_loss, cur_weight = cur_loss_zip.split(":")
            cur_weight = float(cur_weight)
            self.weight_dict[cur_loss.lower()] = cur_weight
            if cur_loss.lower() == "mel":
                self.loss_dict[cur_loss.lower()] = MelLoss(sampling_rate=self.sampling_rate)
            elif cur_loss.lower() == "multi-mel":
                self.loss_dict[cur_loss.lower()] = MultiresolutionMelLoss(
                    sampling_rate=self.sampling_rate
                )
            elif cur_loss.lower() == "multi-stft":
                self.loss_dict[cur_loss.lower()] = MultiresolutionSTFTLoss()
            elif cur_loss.lower() in ["score_mse", "score_mae"]:
                # 这两种直接在 _loss 里用 _reduce_op_4 计算
                continue
            else:
                raise NotImplementedError(f"Unsupported loss type: {cur_loss}")

        # 保存超参便于 ckpt/恢复
        self.save_hyperparameters()

    # ----------------- Optim / EMA -----------------
    def configure_optimizers(self):
        if self.opt_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
            )
        else:
            optimizer = torch.optim.AdamW(
                self.dnn.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
            )
        if self.lr_scheduler_interval == "step":
            if self.lr_tmax_steps > 0:
                t_max = self.lr_tmax_steps
            elif getattr(self, "trainer", None) is not None and getattr(self.trainer, "max_steps", None) not in [None, -1, 0]:
                t_max = int(self.trainer.max_steps)
            elif getattr(self, "trainer", None) is not None and getattr(self.trainer, "estimated_stepping_batches", None) is not None:
                t_max = int(self.trainer.estimated_stepping_batches)
            else:
                t_max = int(self.max_epochs) * 1000
                warnings.warn(
                    f"lr_scheduler_interval=step but total steps is unknown; fallback T_max={t_max}. "
                    f"Consider setting trainer.max_steps or model.lr_tmax_steps."
                )
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.lr_eta_min)
            scheduler_cfg = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.lr_eta_min)
            scheduler_cfg = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}

    def lr_scheduler_step(self, scheduler, metric=None):
        scheduler.step()

    def on_train_epoch_end(self) -> None:
        if self.use_gan:
            self.scheduler_d.step()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # ----------------- EMA checkpoint hooks -----------------
    def on_load_checkpoint(self, checkpoint) -> None:
        """
        使 EMA 在恢复 / 推理由 checkpoint 加载时行为与 ScoreModelGAN 一致：

        - 如果 checkpoint 中包含 'ema' 状态，则正常恢复；
        - 否则置 _error_loading_ema=True，后续 eval()/train() 均不会再尝试覆盖 dnn 参数，
          避免在推理时被未训练的 shadow_params 覆盖成随机权重，导致听起来像白噪音。
        """
        ema_state = checkpoint.get("ema", None)
        if ema_state is not None:
            try:
                self.ema.load_state_dict(ema_state)
            except Exception as e:
                self._error_loading_ema = True
                warnings.warn(f"Failed to load EMA state from checkpoint, disable EMA. Error: {e}")
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint! Disable EMA for this run.")

    def on_save_checkpoint(self, checkpoint) -> None:
        """
        将 EMA 状态一并写入 checkpoint，方便后续恢复 / 推理时使用。
        """
        checkpoint["ema"] = self.ema.state_dict()

    def train(self, mode: bool = True, no_ema: bool = False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                self.ema.store(self.dnn.parameters())
                self.ema.copy_to(self.dnn.parameters())
            else:
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())
        return res

    def eval(self, no_ema: bool = False):
        return self.train(False, no_ema=no_ema)

    def to(self, *args, **kwargs):
        # 确保 EMA 的 shadow params 与模型参数位于同一设备
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # ----------------- Loss helpers -----------------
    def _loss(
        self, err: torch.Tensor, score_wav: torch.Tensor, x_wav: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        err: (B, 2, F, T)
        score_wav: (B, L)
        x_wav: (B, L)
        """
        loss_val_dict: Dict[str, torch.Tensor] = {}
        total = 0.0
        for k in self.loss_type_list:
            k_low = k.lower().split(":")[0] if ":" in k else k.lower()
            if k_low in ["multi-mel", "mel", "multi-stft"]:
                cur_loss = self.loss_dict[k_low](x_wav, score_wav, self._reduce_op_3)
                total = total + self.weight_dict[k_low] * cur_loss
            elif k_low == "score_mse":
                cur_loss = self._reduce_op_4(torch.square(torch.abs(err)))
                total = total + self.weight_dict[k_low] * cur_loss
            elif k_low == "score_mae":
                cur_loss = self._reduce_op_4(torch.abs(err))
                total = total + self.weight_dict[k_low] * cur_loss
            else:
                continue
            loss_val_dict[k_low] = cur_loss.detach()
        return total, loss_val_dict

    # ----------------- Core forward / training step -----------------
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, F, T)
        y: (B, 2, F, T)
        t: (B,)
        """
        return self.dnn(x, cond=y, time_cond=t)

    def _stft_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, L)
        返回 complex STFT: (B, F, T)
        """
        window = self.stft_window
        if window.device != audio.device:
            window = window.to(audio.device)
        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=window,
            center=True,
            return_complex=True,
        )
        return spec

    def _spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        """
        复数谱前向变换：在复杂谱域做幅度压缩，与 SpecsDataModule.spec_fwd / enhancement.py 保持一致。
        """
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1.0:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            pass
        return spec

    def _spec_back(self, spec: torch.Tensor) -> torch.Tensor:
        """
        复数谱反变换：与 _spec_fwd 精确互逆。
        """
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1.0:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1.0 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1.0) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            pass
        return spec

    def _ri_score_to_wav(self, score: torch.Tensor, real_len: int) -> torch.Tensor:
        """
        将压缩谱域中的 score (B,2,F,T) 映射回波形：
        - 先还原为复数压缩谱；
        - 若 drop_last_freq，则补回最后一个频带；
        - 再通过 _spec_back 恢复到原始 STFT，再 iSTFT 得到波形。
        """
        score_complex = torch.complex(score[:, 0], score[:, 1])  # (B, F, T) 或 (B, F-1, T)
        if self.drop_last_freq:
            # onesided STFT 的 Nyquist bin（最后一带）对实信号应为纯实数且通常能量很小。
            # 训练时我们不预测该频带（drop_last_freq），因此此处用 0 填充更稳，
            # 避免复制相邻频带将能量“抬”到 Nyquist 附近，导致 >21.5kHz 噪声。
            last_band = torch.zeros_like(score_complex[:, :1, :]).contiguous()
            score_complex = torch.cat([score_complex, last_band], dim=1)  # (B, F, T)
        score_complex = self._spec_back(score_complex)
        window = self.stft_window
        if window.device != score_complex.device:
            window = window.to(score_complex.device)
        score_wav = torch.istft(
            score_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=window,
            center=True,
            length=real_len,
        )
        return score_wav

    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        batch: dict(audio=(B,1,L), mel=(B,num_mels,frames), f0=(B,frames))
        """
        audio = batch["audio"].to(self.device, non_blocking=True)  # (B, 1, L)
        f0 = batch["f0"].to(self.device, non_blocking=True)  # (B, frames)

        x_audio = audio.squeeze(1)  # (B, L)
        real_len = x_audio.shape[-1]

        # 1) 目标谱 X：由真实音频 STFT 得到
        x_spec = self._stft_audio(x_audio)  # (B, F, T)

        mel = batch.get("mel", None)
        if mel is None:
            mel_mag = torch.matmul(self.mel_basis, x_spec.abs())
            mel = spectral_normalize_torch(mel_mag)
        else:
            mel = mel.to(self.device, non_blocking=True)  # (B, num_mels, frames)

        if self.drop_last_freq:
            x_spec = x_spec[:, :-1].contiguous()
        x_spec = self._spec_fwd(x_spec)
        x_ri = torch.stack([x_spec.real, x_spec.imag], dim=1)  # (B, 2, F, T)

        # 2) 条件谱 Y：由 mel + F0 通过 NSF 源 + BCD 前端构造
        cond_full_ri = self.dnn.build_cond(mel, f0)  # (B, 2, F_full, T)
        if self.drop_last_freq:
            cond_full_ri = cond_full_ri[:, :, :-1].contiguous()

        cond_spec = torch.complex(cond_full_ri[:, 0], cond_full_ri[:, 1])  # (B, F, T) or (B, F-1, T)
        cond_spec = self._spec_fwd(cond_spec)
        cond_full = torch.stack([cond_spec.real, cond_spec.imag], dim=1)  # (B, 2, F, T)

        # 确保时间维对齐
        T_common = min(x_ri.shape[-1], cond_full.shape[-1])
        x_ri = x_ri[..., :T_common]
        cond_full = cond_full[..., :T_common]

        x = x_ri
        y = cond_full

        # 3) BridgeGAN 前向扩散
        if self.sde_name == "bridgegan":
            # 判别器更新（与 ScoreModelGAN 一致，每隔一步）
            if batch_idx % 2 == 0 and self.use_gan:
                t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False)
                t = torch.clamp(t, self.t_eps, 1.0 - self.t_eps)
                xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)
                score = self(xt, t, y)  # generator 输出
                # 将 score 还原为波形，用于判别器（与 bridge-only 相同的谱压缩/反变换流程）
                score_wav = self._ri_score_to_wav(score, real_len)

                self.optim_d.zero_grad()
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(x_audio, score_wav.detach())
                loss_disc_f, _, _ = DiscriminatorLoss(y_df_hat_r, y_df_hat_g)
                y_ds_hat_r, y_ds_hat_g, _, _ = self.mrd(x_audio, score_wav.detach())
                loss_disc_s, _, _ = DiscriminatorLoss(y_ds_hat_r, y_ds_hat_g)
                L_D = loss_disc_s + loss_disc_f
                L_D.backward()
                self.optim_d.step()

            # 生成器 / score 网络更新
            t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False)
            t = torch.clamp(t, self.t_eps, 1.0 - self.t_eps)
            xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)
            score = self(xt, t, y)

            err = score - target

            if len(self.loss_dict) > 0:
                score_wav = self._ri_score_to_wav(score, real_len)

                if self.use_gan:
                    _, y_df_g, fmap_f_r, fmap_f_g = self.mpd(x_audio, score_wav)
                    _, y_ds_g, fmap_s_r, fmap_s_g = self.mrd(x_audio, score_wav)
                    loss_fm_f = FeatureMatchingLoss(fmap_f_r, fmap_f_g)
                    loss_fm_s = FeatureMatchingLoss(fmap_s_r, fmap_s_g)
                    loss_gen_f, _ = GeneratorLoss(y_df_g)
                    loss_gen_s, _ = GeneratorLoss(y_ds_g)
                    L_GAN_G = loss_gen_s + loss_gen_f
                    L_FM = loss_fm_s + loss_fm_f
                else:
                    L_GAN_G, L_FM = 0.0, 0.0
            else:
                score_wav = x_audio
                L_GAN_G, L_FM = 0.0, 0.0

            loss, loss_val_dict = self._loss(err, score_wav=score_wav, x_wav=x_audio)
            loss_total = loss + 15.0 * (L_GAN_G + L_FM)
            return loss_total, loss_val_dict

        raise NotImplementedError(f"SDE {self.sde_name} not supported in NsfBridgeScoreModel.")

    # ----------------- Lightning hooks -----------------
    def training_step(self, batch, batch_idx):
        loss, loss_val_dict = self._step(batch, batch_idx)
        for k, v in loss_val_dict.items():
            self.log(
                f"train_loss_{k}",
                v,
                on_step=True,
                on_epoch=False,
                batch_size=batch["audio"].shape[0],
                prog_bar=False,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证阶段不更新判别器，只复用生成器 / score 的前向与损失计算。
        # 传入 batch_idx=1 可避免 _step 中判别器更新分支被触发（与 ScoreModelGAN 保持一致）。
        loss, loss_val_dict = self._step(batch, 1)
        for k, v in loss_val_dict.items():
            self.log(
                f"valid_loss_{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=batch["audio"].shape[0],
                prog_bar=False,
            )
        # 仅在第一个 batch 上做可视化，避免开销过大
        if batch_idx == 0 and hasattr(self.logger, "experiment"):
            try:
                tb = self.logger.experiment
                import torch.nn.functional as F
                import matplotlib.pyplot as plt

                audio = batch["audio"].to(self.device, non_blocking=True)  # (B,1,L)
                f0 = batch["f0"].to(self.device, non_blocking=True)
                x_audio = audio.squeeze(1)  # (B,L)

                # 重新构造一次 score_wav，用于可视化（不影响损失）
                with torch.no_grad():
                    # 目标谱 X
                    x_spec = self._stft_audio(x_audio)  # (B,F,T)

                    mel = batch.get("mel", None)
                    if mel is None:
                        mel_mag = torch.matmul(self.mel_basis, x_spec.abs())
                        mel = spectral_normalize_torch(mel_mag)
                    else:
                        mel = mel.to(self.device, non_blocking=True)

                    if self.drop_last_freq:
                        x_spec = x_spec[:, :-1].contiguous()
                    x_spec = self._spec_fwd(x_spec)
                    x_ri = torch.stack([x_spec.real, x_spec.imag], dim=1)

                    # 条件谱 Y（同样在压缩谱空间）
                    cond_full_ri = self.dnn.build_cond(mel, f0)
                    if self.drop_last_freq:
                        cond_full_ri = cond_full_ri[:, :, :-1].contiguous()
                    cond_spec = torch.complex(cond_full_ri[:, 0], cond_full_ri[:, 1])
                    cond_spec = self._spec_fwd(cond_spec)
                    cond_full = torch.stack([cond_spec.real, cond_spec.imag], dim=1)

                    T_common = min(x_ri.shape[-1], cond_full.shape[-1])
                    x_ri = x_ri[..., :T_common]
                    cond_full = cond_full[..., :T_common]

                    # BridgeGAN 扩散 + score 预测（同训练）
                    t = torch.rand(x_ri.shape[0], dtype=x_ri.dtype, device=x_ri.device, requires_grad=False)
                    t = torch.clamp(t, self.t_eps, 1.0 - self.t_eps)
                    xt, target = self.sde.forward_diffusion(x0=x_ri, x1=cond_full, t=t)
                    score = self(xt, t, cond_full)

                    real_len = x_audio.shape[-1]
                    score_wav = self._ri_score_to_wav(score, real_len)

                    def _stft_log(spec_audio: torch.Tensor) -> torch.Tensor:
                        """
                        计算单尺度 log10 STFT 频谱，用于可视化。
                        """
                        win_size = self.win_size
                        n_fft = self.n_fft
                        hop = self.hop_size
                        window = self.stft_window
                        if window.device != spec_audio.device:
                            window = window.to(spec_audio.device)
                        y = F.pad(
                            spec_audio.unsqueeze(1),
                            (int((win_size - hop) // 2), int((win_size - hop + 1) // 2)),
                            mode="reflect",
                        ).squeeze(1)
                        spec = torch.stft(
                            y,
                            n_fft,
                            hop_length=hop,
                            win_length=win_size,
                            window=window,
                            center=False,
                            normalized=False,
                            onesided=True,
                            return_complex=True,
                        ).abs()
                        return torch.log10(torch.clamp(spec, min=1e-7))

                    stft_pred = _stft_log(score_wav)
                    stft_gt = _stft_log(x_audio)
                    min_t = min(stft_pred.shape[2], stft_gt.shape[2])
                    stft_pred = stft_pred[:, :, :min_t]
                    stft_gt = stft_gt[:, :, :min_t]
                    spec_cat = torch.cat(
                        [(stft_pred - stft_gt).abs(), stft_gt, stft_pred], dim=2
                    )

                # 画图并写入 TensorBoard（仿照 SingingVocoders）
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.pcolor(spec_cat[0].detach().cpu().numpy().T)
                ax.set_title("NSF-Bridge GT / Pred STFT (log10) and diff")
                ax.set_xlabel("Time")
                ax.set_ylabel("Freq")
                plt.tight_layout()
                tb.add_figure(
                    "validation_nsf_bridge/stft_log10", fig, global_step=self.global_step
                )
                plt.close(fig)

                sr = self.sampling_rate
                tb.add_audio(
                    "validation_nsf_bridge/pred_audio",
                    score_wav[0:1].detach().cpu(),
                    sample_rate=sr,
                    global_step=self.global_step,
                )
                tb.add_audio(
                    "validation_nsf_bridge/gt_audio",
                    x_audio[0:1].detach().cpu(),
                    sample_rate=sr,
                    global_step=self.global_step,
                )
            except Exception as e:
                print(f"[WARN] NSF-Bridge validation visualization failed: {e}")

        return loss
