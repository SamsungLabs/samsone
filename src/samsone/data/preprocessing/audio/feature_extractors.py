from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from transformers import AutoFeatureExtractor

from samsone.data.constants import (
    ATTENTION_MASK_KEY,
    INPUT_FEATURES_KEY,
    INPUT_VALUES_KEY,
)
from samsone.models.audio.poolers import FixedLengthAudioEmbeddingsPooler


def get_features_key(features_keys: list[str]):
    """
    Returns:
        Key of the audio features created by the FeatureExtractor
    """
    return (
        INPUT_FEATURES_KEY if INPUT_FEATURES_KEY in features_keys else INPUT_VALUES_KEY
    )


class FeatureExtractor(nn.Module, ABC):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    @abstractmethod
    def forward(self, x, attention_mask):
        pass


class MelFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: int,
        f_max: int,
        n_mels: int,
        power: int = 2,
        center: bool = True,
        log: bool = False,
        log_eps: float = 1e-7,
    ):
        super().__init__(sample_rate)
        self.mel_features = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            center=center,
        )
        self.log = log
        self.log_eps = log_eps

    def forward(self, x, attention_mask):
        mels = self.mel_features(x).permute(
            0, 2, 1
        )  # output_shape: [batch, time, features]

        if self.log:
            mels = torch.log(torch.clamp(mels, min=self.log_eps))

        output_seq_len = mels.shape[1]

        pooled_attention_mask = FixedLengthAudioEmbeddingsPooler.pool_to_fixed_len(
            attention_mask, output_seq_len, pool_mode="max"
        )

        return {
            INPUT_FEATURES_KEY: mels,
            ATTENTION_MASK_KEY: pooled_attention_mask,
        }


class HuggingFaceFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name_or_path: str, sample_rate: int):
        super().__init__(sample_rate)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path
        )

        if self.sample_rate != self.feature_extractor.sampling_rate:
            raise ValueError(
                f"File sampling rate ({self.sample_rate}) != Feature extractor sampling rate ({self.feature_extractor.sampling_rate})"
            )

    def forward(x, attention_mask):
        pass


class ASTFeatureExtractor(HuggingFaceFeatureExtractor):
    def forward(self, x, attention_mask):
        x = x.numpy()
        features = self.feature_extractor(
            x, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        features[ATTENTION_MASK_KEY] = torch.ones(
            features[INPUT_VALUES_KEY].shape[:2],
            device=features[INPUT_VALUES_KEY].device,
        )

        output_seq_len = features[INPUT_VALUES_KEY].shape[1]

        pooled_attention_mask = FixedLengthAudioEmbeddingsPooler.pool_to_fixed_len(
            attention_mask, output_seq_len, pool_mode="max"
        )

        features[ATTENTION_MASK_KEY] = pooled_attention_mask

        return features


class WhisperFeatureExtractor(HuggingFaceFeatureExtractor):
    def forward(self, x, attention_mask):
        x = x.numpy()
        features = self.feature_extractor(
            x,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        output_seq_len = features[INPUT_FEATURES_KEY].shape[2]

        pooled_attention_mask = FixedLengthAudioEmbeddingsPooler.pool_to_fixed_len(
            attention_mask, output_seq_len, pool_mode="max"
        )

        features[ATTENTION_MASK_KEY] = pooled_attention_mask

        return features


class RawAudioLengthAwareFeatureExtractor(FeatureExtractor):
    def forward(self, x, attention_mask):
        features = {INPUT_VALUES_KEY: x, ATTENTION_MASK_KEY: attention_mask}

        return features
