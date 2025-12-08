import pytorch_lightning as pl

from div.nsf.module import NsfHifiGanModel


class NsfBridgeVocModel(NsfHifiGanModel):
    """
    NSF-BridgeVoc 骨架模型。

    当前阶段：
    - 直接复用 NSF-HiFiGAN 的生成器与判别器结构；
    - 仅通过数据管线改为使用离线 F0（NsfBridgeDataset），并保留统一的损失函数；
    - 为后续替换为 \"NSF 源 + BridgeVoC 子带网络\" 提供独立入口，不影响原始 NSF-HiFiGAN 训练。
    """

    # 目前不需要额外改动，所有逻辑继承自 NsfHifiGanModel
    pass

