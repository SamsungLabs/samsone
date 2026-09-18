from dataclasses import dataclass

import torch.nn as nn


@dataclass
class AudioDataConfig:
    data_dir: str
    wav_processor: nn.Module
    feature_extractor: nn.Module
    sample_rate: int
