from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from samsone.data.constants import ATTENTION_MASK_KEY
from samsone.data.datasets.configs import AudioDataConfig
from samsone.data.preprocessing.audio import get_features_key


@dataclass
class AudioFeatureData:
    audio_features: dict
    audio_wav: torch.Tensor


class AudioFeatureDataset(Dataset):
    def __init__(self, cfg: AudioDataConfig, file_paths: list[list[str]]):
        self.data_dir = Path(cfg.data_dir)
        self.wav_processor = cfg.wav_processor
        self.feature_extractor = cfg.feature_extractor
        self.sample_rate = cfg.sample_rate

        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        audio_files_for_item = self.file_paths[index]
        processed_audios_list = []
        features_list = []

        for file_path_str in audio_files_for_item:
            filename = self.data_dir / file_path_str
            decoder = AudioDecoder(filename, sample_rate=self.sample_rate)
            audio = decoder.get_all_samples().data
            attention_mask = torch.ones_like(audio, device=audio.device)

            processed_audio, attention_mask = self.wav_processor(audio, attention_mask)
            features = self.feature_extractor(processed_audio, attention_mask)

            processed_audios_list.append(processed_audio)
            features_list.append(features)

        # shape: [num_audios, channels, time]
        processed_audio = torch.stack(processed_audios_list, dim=0)

        combined_audio_features = {}
        features_keys = features_list[0].keys()
        for key in features_keys:
            tensors_list = [features[key] for features in features_list]
            combined_audio_features[key] = torch.stack(tensors_list, dim=0)

        audio_features_key = get_features_key(features_keys)

        combined_audio_features[audio_features_key] = combined_audio_features[
            audio_features_key
        ].squeeze(dim=1)  # shape: [num_audios, seq_len, feature_dim]

        combined_audio_features[ATTENTION_MASK_KEY] = combined_audio_features[
            ATTENTION_MASK_KEY
        ].squeeze(dim=1)

        return AudioFeatureData(
            audio_features=combined_audio_features, audio_wav=processed_audio
        )
