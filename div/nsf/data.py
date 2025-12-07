import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from div.data_module import load_wav, mel_spectrogram
from torchfcpe import spawn_bundled_infer_model


class NsfDataset(Dataset):
    """
    基于 wav 列表按需提取 mel 与 f0 的简单数据集。

    - 文件列表格式复用 BridgeVoC 当前的 LibriTTS 风格：每行 \"rel/path|dummy\"。
    - f0 使用 torchfcpe 在线提取，高质量且速度较快。
    """

    def __init__(
        self,
        filelist_path: str,
        raw_wav_root: str,
        sampling_rate: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        num_mels: int,
        num_frames: int,
        subset: str = "train",
        use_gpu_fcpe: bool = False,
    ):
        super().__init__()
        self.raw_wav_root = raw_wav_root
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.num_frames = num_frames
        self.subset = subset
        # 如果开启 GPU F0，并且当前环境有 CUDA，则使用 GPU；否则退回 CPU
        if use_gpu_fcpe and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        with open(filelist_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        self.wav_files: List[str] = []
        for l in lines:
            rel = l.split("|")[0]
            if not rel.endswith(".wav"):
                rel = f"{rel}.wav"
            self.wav_files.append(os.path.join(raw_wav_root, rel))

        if subset == "train":
            random.seed(3407)
            random.shuffle(self.wav_files)

        # 延迟初始化 torchfcpe 模型，避免在未使用时加载权重
        self._fcpe_model = None

    def __len__(self) -> int:
        return len(self.wav_files)

    def _compute_f0(self, audio: np.ndarray, n_frames: int) -> np.ndarray:
        """
        使用 torchfcpe 进行 F0 提取，并对长度进行裁剪/补齐到 n_frames。
        """
        if self._fcpe_model is None:
            # 只在第一次调用时加载一次预训练推理模型
            self._fcpe_model = spawn_bundled_infer_model(device=self.device)

        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(self.device)
        target_len = n_frames
        with torch.no_grad():
            f0_t = self._fcpe_model.infer(
                audio_t,
                sr=self.sampling_rate,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=80,
                f0_max=880,
                interp_uv=False,
                output_interp_target_length=target_len,
            )

        # torchfcpe 返回 (B, T) 或 (T,)；统一转换为 numpy 一维
        if isinstance(f0_t, torch.Tensor):
            f0_np = f0_t.squeeze().cpu().numpy().astype(np.float32)
        else:
            f0_np = np.asarray(f0_t, dtype=np.float32)

        if f0_np.shape[0] >= n_frames:
            f0_np = f0_np[:n_frames]
        else:
            pad = n_frames - f0_np.shape[0]
            f0_np = np.pad(f0_np, (0, pad), mode="edge")
        return f0_np

    def __getitem__(self, idx: int):
        filename = self.wav_files[idx]
        audio = load_wav(filename, self.sampling_rate)  # (T,)
        audio = np.asarray(audio, dtype=np.float32)

        # 计算 mel：返回 (n_mels, frames)
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        mel = mel_spectrogram(
            audio_t,
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
            center=True,
            in_dataset=True,
        )[0]

        total_frames = mel.shape[1]
        if self.num_frames > 0 and total_frames > self.num_frames:
            if self.subset == "train":
                start = random.randint(0, total_frames - self.num_frames)
            else:
                start = 0
            end = start + self.num_frames
        else:
            start = 0
            end = total_frames
        mel = mel[:, start:end]
        frames = mel.shape[1]

        # 对应裁剪 audio
        start_wav = start * self.hop_size
        end_wav = min(len(audio), end * self.hop_size)
        audio_seg = audio[start_wav:end_wav]
        target_len = frames * self.hop_size
        if len(audio_seg) < target_len:
            audio_seg = np.pad(audio_seg, (0, target_len - len(audio_seg)), mode="constant")

        # 提取 f0
        f0 = self._compute_f0(audio_seg, frames)

        return {
            "audio": torch.from_numpy(audio_seg).unsqueeze(0),
            "mel": mel,  # (n_mels, frames)
            "f0": torch.from_numpy(f0),
        }


def create_nsf_dataloaders(
    train_list: str,
    val_list: str,
    raw_wav_root: str,
    sampling_rate: int,
    n_fft: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    num_mels: int,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    use_gpu_fcpe: bool = False,
):
    train_dataset = NsfDataset(
        train_list,
        raw_wav_root,
        sampling_rate,
        n_fft,
        hop_size,
        win_size,
        fmin,
        fmax,
        num_mels,
        num_frames,
        subset="train",
        use_gpu_fcpe=use_gpu_fcpe,
    )
    val_dataset = NsfDataset(
        val_list,
        raw_wav_root,
        sampling_rate,
        n_fft,
        hop_size,
        win_size,
        fmin,
        fmax,
        num_mels,
        num_frames,
        subset="val",
        use_gpu_fcpe=use_gpu_fcpe,
    )

    # 如果在 DataLoader 内使用 GPU 上的 torchfcpe，为避免 fork + CUDA 报错，固定 num_workers=0
    effective_num_workers = 0 if (use_gpu_fcpe and torch.cuda.is_available()) else num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
