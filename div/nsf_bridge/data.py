import os
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from div.data_module import load_wav, mel_spectrogram


class NsfBridgeDataset(Dataset):
    """
    NSF-BridgeVoc 专用数据集：

    - 从 wav 清单中读取音频；
    - 使用离线预计算好的 F0（precompute_f0.py 生成的 .f0.npy）；
    - 在线计算 mel，使 mel / F0 / waveform 在帧级严格对齐；
    - 裁剪固定帧数的片段供模型训练。
    """

    def __init__(
        self,
        filelist_path: str,
        raw_wav_root: str,
        f0_root: str,
        sampling_rate: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        num_mels: int,
        num_frames: int,
        subset: str = "train",
        compute_mel_in_dataset: bool = True,
    ):
        super().__init__()
        self.raw_wav_root = raw_wav_root
        self.f0_root = f0_root
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.num_mels = num_mels
        self.num_frames = num_frames
        self.subset = subset
        self.compute_mel_in_dataset = bool(compute_mel_in_dataset)

        with open(filelist_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        self.rel_paths: List[str] = []
        self.wav_files: List[str] = []
        for l in lines:
            rel = l.split("|")[0].replace("\\", "/")
            if not rel.endswith(".wav"):
                rel = f"{rel}.wav"
            self.rel_paths.append(rel)
            self.wav_files.append(os.path.join(raw_wav_root, rel))

        if subset == "train":
            random.seed(3407)
            paired = list(zip(self.rel_paths, self.wav_files))
            random.shuffle(paired)
            self.rel_paths, self.wav_files = zip(*paired)
            self.rel_paths = list(self.rel_paths)
            self.wav_files = list(self.wav_files)

    def __len__(self) -> int:
        return len(self.wav_files)

    def _load_f0_full(self, rel_path: str) -> np.ndarray:
        """
        从离线 F0 目录中读取完整序列。
        """
        f0_path = os.path.join(self.f0_root, self.subset, rel_path)
        f0_path = f0_path.replace(".wav", ".f0.npy")
        if not os.path.exists(f0_path):
            raise FileNotFoundError(f"F0 file not found for {rel_path}: {f0_path}")
        f0 = np.load(f0_path).astype(np.float32)
        return f0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.wav_files[idx]
        rel = self.rel_paths[idx]

        try:
            audio = load_wav(filename, self.sampling_rate)  # (T,)
        except Exception:
            new_idx = (idx + 1) % len(self.wav_files)
            return self.__getitem__(new_idx)

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        f0_full = self._load_f0_full(rel)

        if self.compute_mel_in_dataset:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            mel_full = mel_spectrogram(
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
            total_frames = mel_full.shape[1]
            if self.subset == "train" and self.num_frames > 0 and total_frames < self.num_frames:
                new_idx = (idx + 1) % len(self.wav_files)
                return self.__getitem__(new_idx)
        else:
            total_frames = int(f0_full.shape[0])
            if self.subset == "train" and self.num_frames > 0 and total_frames < self.num_frames:
                new_idx = (idx + 1) % len(self.wav_files)
                return self.__getitem__(new_idx)

        if self.num_frames > 0 and total_frames > self.num_frames:
            if self.subset == "train":
                start = random.randint(0, total_frames - self.num_frames)
            else:
                start = 0
            end = start + self.num_frames
        else:
            start = 0
            end = total_frames

        frames = end - start
        if self.compute_mel_in_dataset:
            mel = mel_full[:, start:end]

        # 对应裁剪 audio
        start_wav = start * self.hop_size
        end_wav = min(len(audio), end * self.hop_size)
        audio_seg = audio[start_wav:end_wav]
        target_len = frames * self.hop_size
        if len(audio_seg) < target_len:
            audio_seg = np.pad(audio_seg, (0, target_len - len(audio_seg)), mode="constant")

        # 裁剪 F0
        if f0_full.shape[0] < end:
            pad = end - f0_full.shape[0]
            f0_full = np.pad(f0_full, (0, pad), mode="edge")
        f0_seg = f0_full[start:end]

        out: Dict[str, torch.Tensor] = {
            "audio": torch.from_numpy(audio_seg).unsqueeze(0),
            "f0": torch.from_numpy(f0_seg.astype(np.float32)),
        }
        if self.compute_mel_in_dataset:
            out["mel"] = mel
        return out


def create_nsf_bridge_dataloaders(
    train_list: str,
    val_list: str,
    raw_wav_root: str,
    f0_root: str,
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
    compute_mel_in_dataset: bool = True,
):
    train_dataset = NsfBridgeDataset(
        train_list,
        raw_wav_root,
        f0_root,
        sampling_rate,
        n_fft,
        hop_size,
        win_size,
        fmin,
        fmax,
        num_mels,
        num_frames,
        subset="train",
        compute_mel_in_dataset=compute_mel_in_dataset,
    )
    val_dataset = NsfBridgeDataset(
        val_list,
        raw_wav_root,
        f0_root,
        sampling_rate,
        n_fft,
        hop_size,
        win_size,
        fmin,
        fmax,
        num_mels,
        num_frames,
        subset="val",
        compute_mel_in_dataset=compute_mel_in_dataset,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader
