import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from div.backbones.bcd import BCD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for BCD high-SR band modes (CPU-only).")
    parser.add_argument("--F", type=int, default=1024, help="Frequency bins after drop_last_freq.")
    parser.add_argument("--T", type=int, default=16, help="Frame length.")
    parser.add_argument("--B", type=int, default=2, help="Batch size.")
    return parser.parse_args()


def _run_case(name: str, **kwargs) -> None:
    B = kwargs.pop("B")
    F = kwargs.pop("F")
    T = kwargs.pop("T")
    model = BCD(
        nblocks=2,
        hidden_channel=64,
        f_kernel_size=5,
        t_kernel_size=5,
        ada_rank=8,
        ada_alpha=8,
        ada_mode="sola",
        mlp_ratio=1,
        decode_type="ri",
        use_adanorm=True,
        causal=False,
        sampling_rate=44100,
        **kwargs,
    )
    x = torch.randn(B, 2, F, T)
    c = torch.randn(B, 2, F, T)
    t = torch.rand(B)
    y = model(x, c, t)
    print(f"{name}: out={tuple(y.shape)}")


def main() -> None:
    args = _parse_args()
    common = dict(B=args.B, F=args.F, T=args.T)
    _run_case("legacy", highsr_band_mode="legacy", **common)
    _run_case(
        "full_uniform",
        highsr_band_mode="full_uniform",
        highsr_freq_bins=args.F,
        highsr_coarse_stride_f=16,
        **common,
    )
    _run_case(
        "ms_16_8_4",
        highsr_band_mode="ms_16_8_4",
        highsr_freq_bins=args.F,
        highsr_coarse_stride_f=16,
        highsr_refine8_start=256,
        highsr_refine4_start=672,
        highsr_refine_overlap=64,
        highsr_refine8_nblocks=1,
        highsr_refine4_nblocks=1,
        **common,
    )


if __name__ == "__main__":
    main()
