import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from div.nsf_bridge import create_nsf_bridge_dataloaders
from div.nsf_bridge.module import NsfBridgeVocModel

try:
    import yaml
except ImportError:
    yaml = None


def _parse_args():
    parser = argparse.ArgumentParser(description="NSF-BridgeVoc 训练入口（使用离线 F0）")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径，例如 configs/nsf_bridgevoc_44k1.yaml",
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

    # 数据加载器（使用离线 F0）
    train_loader, val_loader = create_nsf_bridge_dataloaders(
        train_list=data_cfg["train_data_dir"],
        val_list=data_cfg["val_data_dir"],
        raw_wav_root=data_cfg["raw_wavfile_path"],
        f0_root=data_cfg.get("f0_root", "./data/f0"),
        sampling_rate=data_cfg["sampling_rate"],
        n_fft=data_cfg["n_fft"],
        hop_size=data_cfg["hop_size"],
        win_size=data_cfg["win_size"],
        fmin=data_cfg["fmin"],
        fmax=data_cfg["fmax"],
        num_mels=data_cfg["num_mels"],
        num_frames=data_cfg["num_frames"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    # 模型骨架：目前复用 NSF-HiFiGAN 结构，后续可以在 NsfBridgeVocModel 内替换为 nsf+BridgeVoC
    model = NsfBridgeVocModel(
        sampling_rate=data_cfg["sampling_rate"],
        num_mels=data_cfg["num_mels"],
        n_fft=data_cfg["n_fft"],
        hop_size=data_cfg["hop_size"],
        win_size=data_cfg["win_size"],
        fmin=data_cfg["fmin"],
        fmax=data_cfg["fmax"],
        lr=model_cfg.get("lr", 2e-4),
        beta1=model_cfg.get("beta1", 0.8),
        beta2=model_cfg.get("beta2", 0.99),
        upsample_initial_channel=model_cfg.get("upsample_initial_channel", 512),
        upsample_rates=model_cfg.get("upsample_rates", [8, 8, 2, 2, 2]),
        upsample_kernel_sizes=model_cfg.get("upsample_kernel_sizes", [16, 16, 4, 4, 4]),
        resblock=model_cfg.get("resblock", "1"),
        resblock_kernel_sizes=model_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilation_sizes=model_cfg.get(
            "resblock_dilation_sizes",
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        ),
        discriminator_periods=model_cfg.get("discriminator_periods", [2, 3, 5, 7, 11]),
        mini_nsf=model_cfg.get("mini_nsf", False),
        noise_sigma=model_cfg.get("noise_sigma", 0.0),
        loss_fft_sizes=tuple(model_cfg.get("loss_fft_sizes", [512, 1024, 2048])),
        loss_hop_sizes=tuple(model_cfg.get("loss_hop_sizes", [128, 256, 512])),
        loss_win_lengths=tuple(model_cfg.get("loss_win_lengths", [512, 1024, 2048])),
        aux_mel_weight=model_cfg.get("aux_mel_weight", 45.0),
        aux_stft_weight=model_cfg.get("aux_stft_weight", 2.5),
    )

    # 日志与检查点
    log_dir = os.path.join("ckpt", "nsf_bridgevoc_44k1")
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="nsf_bridgevoc",
    )

    ckpt_every_n_steps = trainer_cfg.get("ckpt_every_n_steps", 10000)
    val_check_interval = trainer_cfg.get("val_check_interval", 20000)

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="step={step}",
        save_top_k=-1,
        every_n_train_steps=ckpt_every_n_steps,
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 1000),
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 1),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        logger=logger,
        callbacks=[checkpoint_cb, TQDMProgressBar(refresh_rate=100)],
        val_check_interval=val_check_interval,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

