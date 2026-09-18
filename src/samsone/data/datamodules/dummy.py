from dataclasses import dataclass

import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader

from samsone.data.datasets import (
    CombinedDummyAudioTextDataset,
    DummyAudioDataset,
    DummyTextDataset,
)


@dataclass
class DummySplitConfig:
    audio_time_dim: int
    feature_dim: int
    text_len: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int


class DummyDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_config: DummySplitConfig,
        valid_config: DummySplitConfig,
    ):
        super().__init__()
        self.train_config = train_config
        self.valid_config = valid_config

    def _create_dataloader(self, config: DummySplitConfig):
        audio_dataset = DummyAudioDataset(
            time_dim=config.audio_time_dim,
            feature_dim=config.feature_dim,
            samples_per_epoch=config.samples_per_epoch,
        )

        text_dataset = DummyTextDataset(
            string_length=config.text_len, samples_per_epoch=config.samples_per_epoch
        )

        dataset = CombinedDummyAudioTextDataset(
            audio_dataset=audio_dataset,
            text_dataset=text_dataset,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def train_dataloader(self):
        return self._create_dataloader(self.train_config)

    def val_dataloader(self):
        return self._create_dataloader(self.valid_config)

    def collate_fn(self, batch):
        """
        Collates a batch of data from CombinedAudioTextDataset.
        The batch is a list of tuples: [(audio_data_1, text_data_1), ...]
        audio_data_i is (spec_i, audio_i)
        text_data_i is a list of indices.
        """
        audio_specs, audio_waves, text_indices_list = zip(
            *[(item[0][0], item[0][1], item[1]) for item in batch]
        )

        # Stack tensors for audio data
        # audio_specs: list of (time_dim, feature_dim) -> (batch_size, time_dim, feature_dim)
        # audio_waves: list of (waveform_len,) -> (batch_size, waveform_len)
        audio_specs_batch = torch.stack(audio_specs)
        audio_waves_batch = torch.stack(audio_waves)

        # Pad text indices
        # text_indices_list: list of lists of indices
        max_len_in_batch = (
            max(len(indices) for indices in text_indices_list)
            if text_indices_list
            else 0
        )

        padded_text_indices = []
        for indices in text_indices_list:
            if len(indices) < max_len_in_batch:
                padded_indices = indices + [0] * (max_len_in_batch - len(indices))
            else:
                padded_indices = indices[:max_len_in_batch]
            padded_text_indices.append(torch.tensor(padded_indices, dtype=torch.long))

        # padded_text_indices: list of (seq_len,) -> (batch_size, seq_len)
        text_indices_batch = torch.stack(padded_text_indices)

        return (audio_specs_batch, audio_waves_batch), text_indices_batch
