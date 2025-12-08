import math
from typing import Any

import torch
import torch.nn as nn

from .shared import BackboneRegistry
from .bcd import BCD
from div.data_module import inverse_mel, mel_spectrogram
from div.nsf.nsf_hifigan import SourceModuleHnNSF


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
        )

        # STFT window 用于从 NSF 激励波形提取相位
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

        # 4) 对齐时间维度（通常 T_src ≈ T_mel）
        T_common = min(spec_src.shape[-1], mag_mel.shape[-1])
        spec_src = spec_src[..., :T_common]
        pha_src = pha_src[..., :T_common]
        mag_mel = mag_mel[..., :T_common]

        # 5) 组合 NSF 相位 + mel 幅度，作为条件谱
        cond_spec = torch.complex(
            mag_mel * torch.cos(pha_src), mag_mel * torch.sin(pha_src)
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

