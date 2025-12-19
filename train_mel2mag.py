import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# 确保无论从哪个工作目录运行，本工程根目录都在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from div.nsf_bridge import create_nsf_bridge_dataloaders
from div.nsf_bridge.mel2mag_model import Mel2MagLightning

try:
    import yaml
except ImportError:
    yaml = None


def _parse_args():
    parser = argparse.ArgumentParser(description="Mel2Mag 预训练入口（mel(+f0)->linear STFT mag）")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径，例如 configs/mel2mag_44k1.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None, help="可选：checkpoint 路径。")
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help="仅加载模型权重，不恢复优化器/调度器状态；用于继续训练或迁移。",
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="加载权重时使用 strict=True（默认 strict=False，会自动忽略缺失 key 与 shape 不匹配）。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if yaml is None:
        raise ImportError("需要 PyYAML 才能解析配置文件，请先安装：pip install pyyaml")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer_cfg = cfg.get("trainer", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    train_loader, val_loader = create_nsf_bridge_dataloaders(
        train_list=data_cfg["train_data_dir"],
        val_list=data_cfg["val_data_dir"],
        raw_wav_root=data_cfg["raw_wavfile_path"],
        f0_root=data_cfg.get("f0_root", "./data/f0"),
        sampling_rate=data_cfg["sampling_rate"],
        n_fft=data_cfg["n_fft"],
        hop_size=data_cfg["hop_size"],
        win_size=data_cfg["win_size"],
        fmin=data_cfg.get("fmin", 0.0),
        fmax=data_cfg.get("fmax", None),
        num_mels=data_cfg["num_mels"],
        num_frames=data_cfg["num_frames"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        compute_mel_in_dataset=data_cfg.get("compute_mel_in_dataset", True),
    )

    model = Mel2MagLightning(
        sampling_rate=data_cfg["sampling_rate"],
        n_fft=data_cfg["n_fft"],
        hop_size=data_cfg["hop_size"],
        win_size=data_cfg["win_size"],
        num_mels=data_cfg["num_mels"],
        drop_last_freq=model_cfg.get("drop_last_freq", True),
        hidden=model_cfg.get("hidden", 256),
        n_blocks=model_cfg.get("n_blocks", 6),
        kernel_size=model_cfg.get("kernel_size", 5),
        dropout=model_cfg.get("dropout", 0.0),
        f0_max=model_cfg.get("f0_max", 1100.0),
        lr=model_cfg.get("lr", 2e-4),
        beta1=model_cfg.get("beta1", 0.8),
        beta2=model_cfg.get("beta2", 0.99),
        opt_type=model_cfg.get("opt_type", "AdamW"),
        hf_fmin_hz=model_cfg.get("hf_fmin_hz", 6000.0),
        hf_fmax_hz=model_cfg.get("hf_fmax_hz", 15000.0),
        hf_weight=model_cfg.get("hf_weight", 1.0),
        hf_gate_ratio=model_cfg.get("hf_gate_ratio", 0.01),
        edge_weight=model_cfg.get("edge_weight", 0.0),
        eps=model_cfg.get("eps", 1e-6),
    )

    ckpt_path_for_trainer = args.ckpt_path
    if args.ckpt_path and args.weights_only:
        import torch

        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = {}
        skipped = 0
        for k, v in state_dict.items():
            if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped += 1
        missing, unexpected = model.load_state_dict(filtered, strict=args.strict_load)
        print(f"[INFO] weights_only load: loaded={len(filtered)}, skipped={skipped}, missing={len(missing)}, unexpected={len(unexpected)}")
        ckpt_path_for_trainer = None

    base_log_dir = trainer_cfg.get("log_dir", os.path.join("ckpt", "mel2mag"))
    run_name = trainer_cfg.get("run_name", "mel2mag")
    logger = TensorBoardLogger(save_dir=base_log_dir, name=run_name)
    exp_dir = os.path.join(base_log_dir, run_name)

    ckpt_every_n_steps = trainer_cfg.get("ckpt_every_n_steps", 10000)
    val_check_interval = trainer_cfg.get("val_check_interval", 20000)
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="step={step}",
        save_top_k=-1,
        every_n_train_steps=ckpt_every_n_steps,
        save_on_train_epoch_end=False,
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 1000),
        max_steps=trainer_cfg.get("max_steps", -1),
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 1),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor_cb, TQDMProgressBar(refresh_rate=100)],
        val_check_interval=val_check_interval,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path_for_trainer)


if __name__ == "__main__":
    main()

