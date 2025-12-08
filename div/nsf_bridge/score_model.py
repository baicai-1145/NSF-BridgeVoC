from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        )

        # GAN 判别器（与 ScoreModelGAN 相同）
        self.opt_type = opt_type
        self.use_gan = use_gan
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_epochs = max_epochs
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
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def lr_scheduler_step(self, scheduler, metric=None):
        scheduler.step()
        if self.use_gan:
            self.scheduler_d.step()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

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
        loss_val_dict: Dict[str, float] = {}
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
            loss_val_dict[k_low] = float(cur_loss.item())
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
        window = torch.hann_window(self.win_size).to(audio.device)
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

    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        batch: dict(audio=(B,1,L), mel=(B,num_mels,frames), f0=(B,frames))
        """
        audio = batch["audio"].to(self.device)  # (B, 1, L)
        mel = batch["mel"].to(self.device)  # (B, num_mels, frames)
        f0 = batch["f0"].to(self.device)  # (B, frames)

        x_audio = audio.squeeze(1)  # (B, L)
        real_len = x_audio.shape[-1]

        # 1) 目标谱 X：由真实音频 STFT 得到
        x_spec = self._stft_audio(x_audio)  # (B, F, T)
        if self.drop_last_freq:
            x_spec = x_spec[:, :-1].contiguous()
        x_ri = torch.stack([x_spec.real, x_spec.imag], dim=1)  # (B, 2, F, T)

        # 2) 条件谱 Y：由 mel + F0 通过 NSF 源 + BCD 前端构造
        cond_full = self.dnn.build_cond(mel, f0)  # (B, 2, F_full, T)
        if self.drop_last_freq:
            cond_full = cond_full[:, :, :-1].contiguous()

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
                t = torch.clamp(t, self.sde.offset, 1.0 - self.sde.offset)
                xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)
                score = self(xt, t, y)  # generator 输出

                # 将 score 还原为波形，用于判别器
                score_mag = torch.norm(score, dim=1)  # (B, F, T)
                if self.drop_last_freq:
                    last_mag = score_mag[:, -1, None]
                    score_mag_ = torch.cat([score_mag, last_mag], dim=1)
                else:
                    score_mag_ = score_mag
                score_pha = torch.atan2(score[:, -1], score[:, 0])
                if self.drop_last_freq:
                    last_pha = score_pha[:, -1, None]
                    score_pha_ = torch.cat([score_pha, last_pha], dim=1)
                else:
                    score_pha_ = score_pha

                score_decom = torch.complex(
                    score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_)
                )
                score_wav = torch.istft(
                    score_decom,
                    n_fft=self.n_fft,
                    hop_length=self.hop_size,
                    win_length=self.win_size,
                    center=True,
                    length=real_len,
                )

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
            t = torch.clamp(t, self.sde.offset, 1.0 - self.sde.offset)
            xt, target = self.sde.forward_diffusion(x0=x, x1=y, t=t)
            score = self(xt, t, y)

            err = score - target

            if len(self.loss_dict) > 0:
                score_mag = torch.norm(score, dim=1)  # (B, F, T)
                if self.drop_last_freq:
                    last_mag = score_mag[:, -1, None]
                    score_mag_ = torch.cat([score_mag, last_mag], dim=1)
                else:
                    score_mag_ = score_mag
                score_pha = torch.atan2(score[:, -1], score[:, 0])
                if self.drop_last_freq:
                    last_pha = score_pha[:, -1, None]
                    score_pha_ = torch.cat([score_pha, last_pha], dim=1)
                else:
                    score_pha_ = score_pha

                score_decom = torch.complex(
                    score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_)
                )
                score_wav = torch.istft(
                    score_decom,
                    n_fft=self.n_fft,
                    hop_length=self.hop_size,
                    win_length=self.win_size,
                    center=True,
                    length=real_len,
                )

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
            self.log(f"train_loss_{k}", v, on_step=True, on_epoch=True, batch_size=batch["audio"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证阶段不更新判别器，只复用生成器 / score 的前向与损失计算。
        # 传入 batch_idx=1 可避免 _step 中判别器更新分支被触发（与 ScoreModelGAN 保持一致）。
        loss, loss_val_dict = self._step(batch, 1)
        for k, v in loss_val_dict.items():
            self.log(f"valid_loss_{k}", v, on_step=False, on_epoch=True, batch_size=batch["audio"].shape[0])
        # 仅在第一个 batch 上做可视化，避免开销过大
        if batch_idx == 0 and hasattr(self.logger, "experiment"):
            try:
                tb = self.logger.experiment
                import torch.nn.functional as F
                import matplotlib.pyplot as plt

                audio = batch["audio"].to(self.device)  # (B,1,L)
                mel = batch["mel"].to(self.device)
                f0 = batch["f0"].to(self.device)
                x_audio = audio.squeeze(1)  # (B,L)

                # 重新构造一次 score_wav，用于可视化（不影响损失）
                with torch.no_grad():
                    # 目标谱 X
                    x_spec = self._stft_audio(x_audio)  # (B,F,T)
                    if self.drop_last_freq:
                        x_spec = x_spec[:, :-1].contiguous()
                    x_ri = torch.stack([x_spec.real, x_spec.imag], dim=1)

                    # 条件谱 Y
                    cond_full = self.dnn.build_cond(mel, f0)
                    if self.drop_last_freq:
                        cond_full = cond_full[:, :, :-1].contiguous()
                    T_common = min(x_ri.shape[-1], cond_full.shape[-1])
                    x_ri = x_ri[..., :T_common]
                    cond_full = cond_full[..., :T_common]

                    # BridgeGAN 扩散 + score 预测
                    t = torch.rand(x_ri.shape[0], dtype=x_ri.dtype, device=x_ri.device, requires_grad=False)
                    t = torch.clamp(t, self.sde.offset, 1.0 - self.sde.offset)
                    xt, target = self.sde.forward_diffusion(x0=x_ri, x1=cond_full, t=t)
                    score = self(xt, t, cond_full)

                    score_mag = torch.norm(score, dim=1)  # (B,F,T)
                    if self.drop_last_freq:
                        last_mag = score_mag[:, -1, None]
                        score_mag_ = torch.cat([score_mag, last_mag], dim=1)
                    else:
                        score_mag_ = score_mag
                    score_pha = torch.atan2(score[:, -1], score[:, 0])
                    if self.drop_last_freq:
                        last_pha = score_pha[:, -1, None]
                        score_pha_ = torch.cat([score_pha, last_pha], dim=1)
                    else:
                        score_pha_ = score_pha

                    score_decom = torch.complex(
                        score_mag_ * torch.cos(score_pha_), score_mag_ * torch.sin(score_pha_)
                    )
                    real_len = x_audio.shape[-1]
                    score_wav = torch.istft(
                        score_decom,
                        n_fft=self.n_fft,
                        hop_length=self.hop_size,
                        win_length=self.win_size,
                        center=True,
                        length=real_len,
                    )

                    def _stft_log(spec_audio: torch.Tensor) -> torch.Tensor:
                        """
                        计算单尺度 log10 STFT 频谱，用于可视化。
                        """
                        win_size = self.win_size
                        n_fft = self.n_fft
                        hop = self.hop_size
                        window = torch.hann_window(win_size, device=spec_audio.device)
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
