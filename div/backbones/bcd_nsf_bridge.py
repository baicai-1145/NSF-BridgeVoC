import math
from typing import Any

import torch
import torch.nn as nn

from .shared import BackboneRegistry
from .bcd import BCD
from div.data_module import inverse_mel, mel_spectrogram
from div.nsf.nsf_hifigan import SourceModuleHnNSF
from div.nsf_bridge.mel2mag import Mel2MagConfig, Mel2MagHF


@BackboneRegistry.register("bcd_nsf_bridge")
class NsfBcdBridge(nn.Module):
    """
    NSF 源 + BCD 的 Bridge backbone：

    - 对外接口与 BCD 完全对齐：forward(inpt, cond, time_cond)；
    - 额外提供 build_cond(mel, f0)，用于在计算图内从 mel+F0 构造条件谱 Y。
    """

    @staticmethod
    def add_argparse_args(parser):
        """
        复用 BCD 的超参定义，并补充 NSF 源相关超参。
        """
        # 先注入与 BCD 完全相同的一组参数
        parser = BCD.add_argparse_args(parser)

        # NSF 源相关参数
        parser.add_argument(
            "--harmonic_num",
            type=int,
            default=8,
            help="NSF 谐波数，SourceModuleHnNSF.harmonic_num。",
        )
        parser.add_argument(
            "--sine_amp",
            type=float,
            default=0.1,
            help="NSF 基波正弦的幅度，SourceModuleHnNSF.sine_amp。",
        )
        parser.add_argument(
            "--add_noise_std",
            type=float,
            default=0.003,
            help="NSF 噪声幅度，SourceModuleHnNSF.add_noise_std。",
        )
        parser.add_argument(
            "--voiced_threshold",
            type=float,
            default=0.0,
            help="F0 > voiced_threshold 视为有声，用于 NSF 源的 uv 判定。",
        )
        parser.add_argument(
            "--phase_mask_ratio",
            type=float,
            default=0.1,
            help=(
                "将 NSF 源相位仅应用于其能量较强的频点："
                "mask = mag_src > phase_mask_ratio * max(mag_src, freq). "
                "其余频点相位使用 0，以减少“坑洞/闷/电流声”。"
            ),
        )
        return parser

    def __init__(
        self,
        # BCD 结构参数（与 BCD 保持一致）
        nblocks: int,
        hidden_channel: int,
        f_kernel_size: int,
        t_kernel_size: int,
        ada_rank: int = 16,
        ada_alpha: int = 16,
        ada_mode: str = "sola",
        mlp_ratio: int = 1,
        input_channel: int = 4,
        act_type: str = "gelu",
        pe_type: str = "positional",
        scale: int = 1000,
        decode_type: str = "ri",
        use_adanorm: bool = True,
        causal: bool = False,
        sampling_rate: int = 24000,
        # 频谱 / mel 参数（与 DataModule / NsfBridge 保持一致）
        n_fft: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: float = 0.0,
        fmax: float = 12000.0,
        num_mels: int = 100,
        # NSF 源相关
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        phase_mask_ratio: float = 0.1,
        mel_phase_gate_ratio: float = 0.0,
        # Mel2Mag (optional, for T5.8)
        cond_mag_source: str = "inverse_mel",  # inverse_mel|mel2mag
        mel2mag_ckpt: str | None = None,
        mel2mag_weight: float = 0.0,  # mixed: mag=(1-w)*inv + w*hat
        mel2mag_hidden: int = 256,
        mel2mag_n_blocks: int = 6,
        mel2mag_kernel_size: int = 5,
        mel2mag_dropout: float = 0.0,
        mel2mag_f0_max: float = 1100.0,
        mel2mag_eps: float = 1e-6,
        mel2mag_freeze: bool = True,
        **unused_kwargs: Any,
    ):
        super().__init__()

        # 保存关键配置
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels

        # NSF 谐波源：根据 F0 生成谐波激励
        self.source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=harmonic_num,
            sine_amp=sine_amp,
            add_noise_std=add_noise_std,
            voiced_threshold=voiced_threshold,
        )
        self.voiced_threshold = voiced_threshold
        self.phase_mask_ratio = float(phase_mask_ratio)
        self.mel_phase_gate_ratio = float(mel_phase_gate_ratio)

        self.cond_mag_source = str(cond_mag_source).lower()
        self.mel2mag_weight = float(mel2mag_weight)

        # Mel2MagHF: mel(+f0/uv) -> linear STFT magnitude (optionally loaded from a Lightning ckpt)
        n_freq = int(self.n_fft // 2 + 1)  # build_cond works in full STFT bins (before drop_last_freq)
        cfg = Mel2MagConfig(
            num_mels=int(self.num_mels),
            n_freq=n_freq,
            hidden=int(mel2mag_hidden),
            n_blocks=int(mel2mag_n_blocks),
            kernel_size=int(mel2mag_kernel_size),
            dropout=float(mel2mag_dropout),
            f0_max=float(mel2mag_f0_max),
            eps=float(mel2mag_eps),
        )
        self.mel2mag = Mel2MagHF(cfg)
        if mel2mag_ckpt:
            self.load_mel2mag_ckpt(str(mel2mag_ckpt), strict=False)
        if bool(mel2mag_freeze):
            for p in self.mel2mag.parameters():
                p.requires_grad = False

        # BCD 子带网络作为 STFT 域解码器（结构与 bridge-only 完全一致）
        self.bcd = BCD(
            nblocks=nblocks,
            hidden_channel=hidden_channel,
            f_kernel_size=f_kernel_size,
            t_kernel_size=t_kernel_size,
            mlp_ratio=mlp_ratio,
            ada_rank=ada_rank,
            ada_alpha=ada_alpha,
            ada_mode=ada_mode,
            input_channel=input_channel,
            act_type=act_type,
            pe_type=pe_type,
            scale=scale,
            decode_type=decode_type,
            use_adanorm=use_adanorm,
            causal=causal,
            sampling_rate=sampling_rate,
            **unused_kwargs,
        )

        # STFT window 用于从 NSF 激励波形提取相位
        self.register_buffer(
            "stft_window", torch.hann_window(self.win_size), persistent=False
        )

    def load_mel2mag_ckpt(self, ckpt_path: str, *, strict: bool = False) -> dict:
        """
        Load Mel2MagHF weights from a Mel2MagLightning checkpoint.
        Returns a small report dict for logging/debugging.
        """
        import os

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"mel2mag_ckpt not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        if not isinstance(state, dict):
            raise ValueError("Invalid mel2mag ckpt: missing state_dict")
        filtered = {}
        for k, v in state.items():
            if k.startswith("mel2mag."):
                filtered[k[len("mel2mag.") :]] = v
            elif k in self.mel2mag.state_dict():
                filtered[k] = v
        missing, unexpected = self.mel2mag.load_state_dict(filtered, strict=strict)
        return {
            "ckpt_path": os.path.abspath(ckpt_path),
            "loaded": int(len(filtered)),
            "missing": list(missing),
            "unexpected": list(unexpected),
        }

    def set_mel2mag_weight(self, w: float):
        self.mel2mag_weight = float(w)

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

    def build_cond(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        从 mel + F0 构造条件谱 Y（NSF 相位 + mel 幅度），返回 (B, 2, F, T).

        mel: (B, num_mels, frames)
        f0:  (B, frames)
        """
        B, _, frames = mel.shape

        # 1) NSF 源生成谐波激励（使用 hop_size 作为上采样因子）
        har_source = self.source(f0, self.hop_size)  # (B, T, 1)
        har_source = har_source.transpose(1, 2)  # (B, 1, T)

        # 2) STFT 得到激励的相位
        spec_src = self._stft(har_source)  # (B, F, T_src)
        mag_src = spec_src.abs()
        pha_src = torch.angle(spec_src)

        # 3) cond 幅度谱：inverse_mel(pinverse) 与 Mel2MagHF 的可回退/可混合实现
        # mel 本身已经是 log-mel；inverse_mel 内部会做 exp 反变换
        w = float(self.mel2mag_weight)
        w = max(0.0, min(1.0, w))
        use_hat = (self.cond_mag_source == "mel2mag") or (w > 0.0)
        use_inv = (self.cond_mag_source == "inverse_mel") or (w < 1.0)

        mag_inv = None
        mag_hat = None
        if use_inv:
            mag_inv = inverse_mel(
                mel,
                n_fft=self.n_fft,
                num_mels=self.num_mels,
                sampling_rate=self.sampling_rate,
                hop_size=self.hop_size,
                win_size=self.win_size,
                fmin=self.fmin,
                fmax=self.fmax,
                in_dataset=False,
            ).abs().clamp_min_(1e-6)  # (B, F_full, T_mel)
        if use_hat:
            uv = (f0 > self.voiced_threshold).to(mel.dtype)
            mag_hat = self.mel2mag(mel, f0, uv).clamp_min(1e-6)  # (B, F_?, T)
            # If mel2mag was trained with drop_last_freq, pad the last bin (Nyquist) with zeros.
            if mag_hat.shape[-2] == (mag_src.shape[-2] - 1):
                last = torch.zeros_like(mag_hat[:, :1, :]).contiguous()
                mag_hat = torch.cat([mag_hat, last], dim=-2)

        if mag_inv is None and mag_hat is None:
            raise ValueError(f"Invalid cond_mag_source: {self.cond_mag_source}")
        if mag_inv is None:
            mag_mel = mag_hat
        elif mag_hat is None:
            mag_mel = mag_inv
        else:
            # Align time dims first, then mix
            Tm = min(int(mag_inv.shape[-1]), int(mag_hat.shape[-1]))
            mag_inv = mag_inv[..., :Tm]
            mag_hat = mag_hat[..., :Tm]
            mag_mel = (1.0 - w) * mag_inv + w * mag_hat

        # 4) 对齐时间维度（通常 T_src ≈ T_mel）
        # 同时考虑 F0/uv 的长度，避免后续 mask 在时间维上 mismatch。
        T_common = min(spec_src.shape[-1], mag_mel.shape[-1], f0.shape[-1])
        mag_src = mag_src[..., :T_common]
        pha_src = pha_src[..., :T_common]
        mag_mel = mag_mel[..., :T_common]

        # 5) 相位注入策略（借鉴 SingingVocoders/NSF 思路）：
        #    NSF 谐波源的频谱非常稀疏，在能量接近 0 的频点上 angle() 近似随机。
        #    若将其“涂满全频”作为 mel 幅度谱的相位，会导致频谱坑洞、闷、嘶嘶/电流感。
        #    因此仅在 NSF 源能量较强的频点使用其相位，其余频点相位置 0。
        #    同时在 unvoiced 帧也不注入相位（相位置 0）。
        if self.phase_mask_ratio > 0.0:
            # per-frame max over freq: (B, 1, T)
            max_mag = mag_src.amax(dim=-2, keepdim=True).clamp_min(1e-8)
            pha_mask = mag_src > (self.phase_mask_ratio * max_mag)  # (B, F, T)
        else:
            pha_mask = torch.ones_like(mag_src, dtype=torch.bool)

        # uv mask: (B, 1, T), unvoiced 不注入相位
        uv = (f0 > self.voiced_threshold).to(mag_src.dtype)  # (B, frames)
        uv = uv[..., :T_common].unsqueeze(1)  # (B, 1, T)
        pha_mask = pha_mask & (uv > 0.5)

        # 进一步用 mag_mel 做门控：避免在幅度很低（谐波间隙/静音）处注入相位。
        # 这允许把 phase_mask_ratio 调得更小以覆盖高次谐波，同时不把相位“涂”到谷底噪声上。
        if self.mel_phase_gate_ratio > 0.0:
            max_mel = mag_mel.amax(dim=-2, keepdim=True).clamp_min(1e-8)  # (B, 1, T)
            mel_gate = mag_mel > (self.mel_phase_gate_ratio * max_mel)
            pha_mask = pha_mask & mel_gate

        pha_used = torch.where(pha_mask, pha_src, torch.zeros_like(pha_src))
        cond_spec = torch.complex(
            mag_mel * torch.cos(pha_used), mag_mel * torch.sin(pha_used)
        )  # (B, F, T)
        cond_ri = torch.stack([cond_spec.real, cond_spec.imag], dim=1)  # (B, 2, F, T)
        return cond_ri

    def forward(
        self,
        inpt: torch.Tensor,
        cond: torch.Tensor,
        time_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bridge 接口与 BCD 完全一致：

        inpt: (B, 2, F, T)，扩散过程中的谱状态 x_t
        cond: (B, 2, F, T)，条件谱 Y（建议由 build_cond(mel, f0) 得到）
        time_cond: (B,)，SDE 时间步

        return: (B, 2, F, T)，预测的 score / 目标谱残差
        """
        return self.bcd(inpt, cond=cond, time_cond=time_cond)
