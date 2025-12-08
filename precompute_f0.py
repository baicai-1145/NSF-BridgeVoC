import argparse
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

from div.data_module import load_wav, mel_spectrogram
from div.nsf.f0_utils import extract_f0_fcpe

try:
    import yaml
except ImportError:
    yaml = None


def _parse_args():
    parser = argparse.ArgumentParser(description="离线预计算 F0（torchfcpe）以供 NSF-BridgeVoC 使用")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置路径，例如 configs/nsf_hifigan_44k1.yaml 或未来的 nsf-bridge 配置",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./data/f0",
        help="F0 保存根目录，结构为 out_root/{train,val}/<相对路径>.f0.npy",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="选择只处理 train、只处理 val 或两者都处理",
    )
    parser.add_argument(
        "--num_examples_to_log",
        type=int,
        default=5,
        help="为前若干条样本打印 F0 统计信息，便于人工快速检查 F0 是否异常",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="F0 预处理进程数，大于 1 时启用多进程（建议 2~4）。",
    )
    return parser.parse_args()


def _load_config(path: str):
    if yaml is None:
        raise ImportError("需要 PyYAML 才能解析配置文件，请先安装：pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_scp_lines(scp_path: str) -> Iterable[str]:
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split("|")[0]
            # 统一为正斜杠，避免 Windows 路径混淆
            rel = rel.replace("\\", "/")
            if not rel.endswith(".wav"):
                rel = f"{rel}.wav"
            yield rel


def _process_one_file(
    split_name: str,
    rel_path: str,
    raw_root: str,
    out_root_split: str,
    sampling_rate: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    num_mels: int,
    device: str,
) -> Tuple[bool, Optional[Tuple[str, int, float, float, float, float]]]:
    """
    单条文件的 F0 预处理逻辑，方便在多进程中复用。
    返回:
      (ok, stats) 其中 stats = (rel_path, frames, voiced_ratio, f0_mean, f0_min, f0_max)
    """
    wav_path = os.path.join(raw_root, rel_path)
    wav_path = wav_path.replace("\\", "/")

    try:
        audio = load_wav(wav_path, sampling_rate)
    except Exception as e:
        print(f"[WARN] 读取失败，跳过 {wav_path}: {e}")
        return False, None

    audio = np.asarray(audio, dtype=np.float32)
    # 略保守的幅度范围，减少 torchfcpe 内部对超界值的告警
    audio = np.clip(audio, -0.95, 0.95)

    # 为了使 F0 与训练/推理时使用的 mel 对齐，这里复用同样的 mel_spectrogram
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel = mel_spectrogram(
            audio_t,
            n_fft=n_fft,
            num_mels=num_mels,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            win_size=win_size,
            fmin=fmin,
            fmax=fmax,
            center=True,
            in_dataset=True,
        )[0]

    frames = mel.shape[1]
    if frames <= 0:
        print(f"[WARN] mel 帧数为 0，跳过 {wav_path}")
        return False, None

    target_len = frames * hop_size
    if audio.shape[0] < target_len:
        pad = target_len - audio.shape[0]
        audio = np.pad(audio, (0, pad), mode="constant")
    else:
        audio = audio[:target_len]

    f0 = extract_f0_fcpe(
        audio,
        sr=sampling_rate,
        n_frames=frames,
        device=device,
        f0_min=80.0,
        f0_max=880.0,
        threshold=0.006,
        interp_uv=False,
    )

    out_path = os.path.join(out_root_split, rel_path)
    out_path = out_path.replace(".wav", ".f0.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, f0)

    voiced_mask = f0 > 0
    voiced_ratio = float(voiced_mask.mean()) if f0.size > 0 else 0.0
    if voiced_mask.any():
        f0_mean = float(f0[voiced_mask].mean())
        f0_min = float(f0[voiced_mask].min())
        f0_max = float(f0[voiced_mask].max())
    else:
        f0_mean = f0_min = f0_max = 0.0

    stats = (rel_path, frames, voiced_ratio, f0_mean, f0_min, f0_max)
    return True, stats


def _process_split(
    split_name: str,
    scp_path: str,
    data_cfg: dict,
    out_root: str,
    num_examples_to_log: int,
    num_workers: int,
):
    raw_root = data_cfg["raw_wavfile_path"]
    sampling_rate = data_cfg["sampling_rate"]
    n_fft = data_cfg["n_fft"]
    hop_size = data_cfg["hop_size"]
    win_size = data_cfg["win_size"]
    fmin = data_cfg.get("fmin", 0.0)
    fmax = data_cfg.get("fmax", sampling_rate / 2)
    num_mels = data_cfg["num_mels"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root_split = os.path.join(out_root, split_name)

    print(f"[INFO] 处理 split={split_name}, scp={scp_path}, 输出根目录={out_root_split}, 设备={device}")

    rel_paths = list(_iter_scp_lines(scp_path))
    count = 0

    if num_workers is None or num_workers <= 1:
        # 单进程串行模式（原行为）
        for rel_path in rel_paths:
            ok, stats = _process_one_file(
                split_name,
                rel_path,
                raw_root,
                out_root_split,
                sampling_rate,
                n_fft,
                hop_size,
                win_size,
                fmin,
                fmax,
                num_mels,
                device,
            )
            if not ok:
                continue
            if count < num_examples_to_log and stats is not None:
                rel_path_s, frames, voiced_ratio, f0_mean, f0_min, f0_max = stats
                print(
                    f"[CHECK] {split_name} {rel_path_s}: frames={frames}, "
                    f"voiced_ratio={voiced_ratio:.3f}, f0_mean={f0_mean:.1f}, "
                    f"f0_min={f0_min:.1f}, f0_max={f0_max:.1f}"
                )
            count += 1
    else:
        # 多进程模式
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"[INFO] 使用多进程提取 F0，num_workers={num_workers}")
        tasks = [
            (
                split_name,
                rel_path,
                raw_root,
                out_root_split,
                sampling_rate,
                n_fft,
                hop_size,
                win_size,
                fmin,
                fmax,
                num_mels,
                device,
            )
            for rel_path in rel_paths
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_process_one_file, *t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    ok, stats = fut.result()
                except Exception as e:
                    print(f"[WARN] 子进程处理样本时出错: {e}")
                    continue
                if not ok:
                    continue
                if count < num_examples_to_log and stats is not None:
                    rel_path_s, frames, voiced_ratio, f0_mean, f0_min, f0_max = stats
                    print(
                        f"[CHECK] {split_name} {rel_path_s}: frames={frames}, "
                        f"voiced_ratio={voiced_ratio:.3f}, f0_mean={f0_mean:.1f}, "
                        f"f0_min={f0_min:.1f}, f0_max={f0_max:.1f}"
                    )
                count += 1

    print(f"[INFO] split={split_name} 处理完成，有效样本数={count}")


def main():
    args = _parse_args()
    cfg = _load_config(args.config)
    data_cfg = cfg.get("data", {})

    if not data_cfg:
        raise ValueError("配置文件中缺少 data 部分，无法获取数据路径与音频参数。")

    split = args.split
    num_workers = args.num_workers
    if split in ("train", "both"):
        train_scp = data_cfg["train_data_dir"]
        _process_split("train", train_scp, data_cfg, args.out_root, args.num_examples_to_log, num_workers)
    if split in ("val", "both"):
        val_scp = data_cfg["val_data_dir"]
        _process_split("val", val_scp, data_cfg, args.out_root, args.num_examples_to_log, num_workers)


if __name__ == "__main__":
    main()
