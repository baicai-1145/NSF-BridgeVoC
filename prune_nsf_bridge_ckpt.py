import argparse
import os

import torch
import pytorch_lightning as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "裁剪 NSF-BridgeVoc 的 Lightning checkpoint，只保留推理必需部分。\n"
            "- 输入: 训练得到的 .ckpt（包含优化器、判别器等完整状态）\n"
            "- 输出: 精简版 .ckpt，只包含生成器 dnn（NsfBcdBridge）和必要超参数，"
            "可直接被 infer_nsf_bridgevoc.py 通过 NsfBridgeScoreModel.load_from_checkpoint(strict=False) 加载用于推理。"
        )
    )
    parser.add_argument(
        "--in_ckpt",
        type=str,
        required=True,
        help="原始 Lightning checkpoint 路径，例如 ckpt/nsf_bridgevoc_44k1/checkpoints/step=step=574000.ckpt",
    )
    parser.add_argument(
        "--out_ckpt",
        type=str,
        required=True,
        help="精简后的 checkpoint 保存路径，例如 ckpt/nsf_bridgevoc_44k1/pruned_step=574000.ckpt",
    )
    parser.add_argument(
        "--keep_gan",
        action="store_true",
        help="是否保留判别器 (mpd/mrd) 的参数。默认不保留，仅保留生成器 dnn。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.in_ckpt):
        raise FileNotFoundError(f"输入 checkpoint 不存在: {args.in_ckpt}")

    print(f"[INFO] Loading checkpoint from {args.in_ckpt}")
    try:
        ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.in_ckpt, map_location="cpu")

    state_dict = ckpt.get("state_dict", {})
    if not state_dict:
        raise RuntimeError("输入 checkpoint 中未找到 'state_dict' 字段。")

    new_state_dict = {}
    dropped_keys = []

    for k, v in state_dict.items():
        # 保留生成器 / backbone：dnn.*
        if k.startswith("dnn."):
            new_state_dict[k] = v
            continue

        # 一般来说 BridgeGAN 没有可训练参数，这里仅防御性保留
        if k.startswith("sde.") or k.startswith("sde_"):
            new_state_dict[k] = v
            continue

        # 根据需要可选择性保留判别器
        if args.keep_gan and ("mpd" in k or "mrd" in k):
            new_state_dict[k] = v
            continue

        # 其它（优化器状态、EMA、判别器等）一律丢弃
        dropped_keys.append(k)

    print(f"[INFO] Kept {len(new_state_dict)} tensors in state_dict, dropped {len(dropped_keys)} tensors.")

    # 只保留推理需要的字段：state_dict + hyper_parameters
    new_ckpt = {
        "state_dict": new_state_dict,
        "hyper_parameters": ckpt.get("hyper_parameters", {}),
        "pytorch-lightning_version": ckpt.get("pytorch-lightning_version", pl.__version__),
    }

    out_dir = os.path.dirname(args.out_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(new_ckpt, args.out_ckpt)
    print(f"[INFO] Saved pruned checkpoint to: {args.out_ckpt}")
    print("[INFO] 推理用法示例：")
    print(
        f"  python infer_nsf_bridgevoc.py --ckpt {args.out_ckpt} "
        "--wav your_input.wav --out test_decode/nsf_bridgevoc_pruned.wav"
    )


if __name__ == "__main__":
    main()
