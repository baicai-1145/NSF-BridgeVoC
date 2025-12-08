import numpy as np
import torch

from torchfcpe import spawn_bundled_infer_model


_FCPE_MODELS = {}


def get_fcpe_model(device: str = "cpu"):
    """
    获取（并缓存）给定 device 上的 torchfcpe 推理模型。
    """
    if device not in _FCPE_MODELS:
        _FCPE_MODELS[device] = spawn_bundled_infer_model(device=device)
    return _FCPE_MODELS[device]


def extract_f0_fcpe(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    device: str = "cpu",
    f0_min: float = 80.0,
    f0_max: float = 880.0,
    threshold: float = 0.006,
    interp_uv: bool = False,
) -> np.ndarray:
    """
    使用 torchfcpe 提取 F0，并对长度裁剪/补齐到 n_frames。

    - audio: 一维 numpy 数组，采样率为 sr。
    - 返回: 形状为 (n_frames,) 的 numpy 数组。
    """
    audio = np.asarray(audio, dtype=np.float32)
    # torchfcpe 期望输入在 [-1, 1] 范围，这里稍微保守一些留出浮点/滤波裕量
    audio = np.clip(audio, -0.95, 0.95)

    model = get_fcpe_model(device=device)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        f0_t = model.infer(
            audio_t,
            sr=sr,
            decoder_mode="local_argmax",
            threshold=threshold,
            f0_min=f0_min,
            f0_max=f0_max,
            interp_uv=interp_uv,
            output_interp_target_length=n_frames,
        )

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
