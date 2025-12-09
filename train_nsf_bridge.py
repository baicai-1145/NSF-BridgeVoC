import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# 确保无论从哪个工作目录运行，本工程根目录都在 sys.path 中，
# 这样就不需要每次手动设置 PYTHONPATH=.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from div.nsf_bridge import create_nsf_bridge_dataloaders
from div.nsf_bridge.score_model import NsfBridgeScoreModel

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
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="可选：从已有 checkpoint 恢复/微调的路径，例如 ckpt/nsf_bridgevoc_44k1/checkpoints/step=step=230000.ckpt",
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

    # 模型：NSF 源 + BCD 作为 Bridge backbone，使用 Score-based BridgeGAN 训练
    model = NsfBridgeScoreModel(
        # 频谱 / 数据参数
        sampling_rate=data_cfg["sampling_rate"],
        n_fft=data_cfg["n_fft"],
        hop_size=data_cfg["hop_size"],
        win_size=data_cfg["win_size"],
        fmin=data_cfg["fmin"],
        fmax=data_cfg["fmax"],
        num_mels=data_cfg["num_mels"],
        spec_factor=model_cfg.get("spec_factor", 0.33),
        spec_abs_exponent=model_cfg.get("spec_abs_exponent", 0.5),
        transform_type=model_cfg.get("transform_type", "exponent"),
        drop_last_freq=model_cfg.get("drop_last_freq", True),
        # 优化 / 损失（如果未在 model 段显式指定，则使用合理默认值）
        opt_type=model_cfg.get("opt_type", "AdamW"),
        lr=model_cfg.get("lr", 5e-4),
        beta1=model_cfg.get("beta1", 0.8),
        beta2=model_cfg.get("beta2", 0.99),
        ema_decay=model_cfg.get("ema_decay", 0.999),
        t_eps=model_cfg.get("t_eps", 0.03),
        loss_type_list=model_cfg.get(
            "loss_type_list", "score_mse:1.0,multi-mel:0.4,multi-stft:0.2"
        ),
        use_gan=model_cfg.get("use_gan", True),
        num_eval_files=model_cfg.get("num_eval_files", 20),
        max_epochs=trainer_cfg.get("max_epochs", 1000),
        # SDE / BridgeGAN 参数（如未提供，沿用 default_bridgevoc_44k1.yaml 中设置）
        beta_min=model_cfg.get("beta_min", 0.01),
        beta_max=model_cfg.get("beta_max", 20.0),
        c=model_cfg.get("c", 0.4),
        k=model_cfg.get("k", 2.6),
        bridge_type=model_cfg.get("bridge_type", "gmax"),
        N=model_cfg.get("N", 4),
        offset=model_cfg.get("offset", 1e-5),
        predictor=model_cfg.get("predictor", "x0"),
        sampling_type=model_cfg.get("sampling_type", "sde_first_order"),
        # BCD 结构参数（与 default_bridgevoc_44k1.yaml 中 BackboneScore 保持一致）
        nblocks=model_cfg.get("nblocks", 8),
        hidden_channel=model_cfg.get("hidden_channel", 256),
        f_kernel_size=model_cfg.get("f_kernel_size", 9),
        t_kernel_size=model_cfg.get("t_kernel_size", 11),
        mlp_ratio=model_cfg.get("mlp_ratio", 1),
        ada_rank=model_cfg.get("ada_rank", 32),
        ada_alpha=model_cfg.get("ada_alpha", 32),
        ada_mode=model_cfg.get("ada_mode", "sola"),
        act_type=model_cfg.get("act_type", "gelu"),
        pe_type=model_cfg.get("pe_type", "positional"),
        scale=model_cfg.get("scale", 1000),
        decode_type=model_cfg.get("decode_type", "ri"),
        use_adanorm=model_cfg.get("use_adanorm", True),
        causal=model_cfg.get("causal", False),
        # NSF 源参数：如未显式给出，则使用 SourceModuleHnNSF 默认值
        harmonic_num=model_cfg.get("harmonic_num", 8),
        sine_amp=model_cfg.get("sine_amp", 0.1),
        add_noise_std=model_cfg.get("add_noise_std", 0.003),
        voiced_threshold=model_cfg.get("voiced_threshold", 0.0),
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

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
