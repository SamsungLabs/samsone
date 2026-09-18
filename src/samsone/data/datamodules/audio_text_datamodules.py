from dataclasses import dataclass

import lightning.pytorch as L
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from samsone.data.constants import (
    AUDIO_ORDER_IN_SAMPLE_KEY,
    AUDIO_TO_SAMPLE_ASSIGNMENT_KEY,
    AUDIO_WAV_KEY,
    META_FILEPATHS_KEY,
    META_ID_KEY,
    META_PROMPT_KEY,
    META_RESPONSE_KEY,
)
from samsone.data.datasets import (
    ALMEvalDataset,
    AudioDataConfig,
    AudioFeatureDataset,
    AudioQATextData,
    AudioQATextDataset,
    QATextDataset,
    TextDataConfig,
)
from samsone.data.datasets.concat import WeightedConcatDataset
from samsone.data.preprocessing.audio import get_features_key
from samsone.evaluation import ALMEvaluator


@dataclass
class AudioTextSplitConfig:
    audio_cfg: AudioDataConfig
    text_cfg: TextDataConfig
    meta_paths: list[str]
    batch_size: int
    num_workers: int
    samples_per_epoch: int | None = None
    dataset_weights: list[int] | None = None


class AudioQATextDatamodule(L.LightningDataModule):
    def __init__(
        self,
        train_cfg: AudioTextSplitConfig,
        valid_cfg: AudioTextSplitConfig,
        test_cfgs: list[AudioTextSplitConfig] | None = None,
        evaluators: list[ALMEvaluator] | None = None,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.valid_cfg = valid_cfg
        self.test_cfgs = test_cfgs
        self.evaluators = evaluators

        self.tokenizer = AutoTokenizer.from_pretrained(
            train_cfg.text_cfg.tokenizer_hf_path
        )
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})

    def setup(self, stage=None):
        if stage == "fit":
            assert self.train_cfg.samples_per_epoch, (
                "Set the number of samples per training epoch in config!"
            )

            if not self.valid_cfg.samples_per_epoch:
                self.valid_cfg.samples_per_epoch = (
                    self.train_cfg.samples_per_epoch // 20
                )

            self.train_dataset = self._build_concat_dataset(
                self.train_cfg, weighted=True
            )
            self.valid_dataset = self._build_concat_dataset(
                self.valid_cfg, weighted=True
            )

        if stage == "test":
            if self.test_cfgs and self.evaluators:
                assert len(self.test_cfgs) == len(self.evaluators), (
                    f"The number of test datasets ({len(self.test_cfgs)}) and evaluators ({len(self.evaluators)}) has to be equal!"
                )

                for c in self.test_cfgs:
                    assert not c.dataset_weights, "Test dataset cannot contain weights!"

                self.test_datasets = self._build_test_datasets()
            else:
                raise ValueError(
                    "Cannot perform evaluation: test config or evaluators are not provided."
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            collate_fn=self._custom_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_cfg.batch_size,
            num_workers=self.valid_cfg.num_workers,
            collate_fn=self._custom_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        dataloaders = []

        for ds, cfg in zip(self.test_datasets, self.test_cfgs):
            dl = DataLoader(
                dataset=ds,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                collate_fn=self._custom_collate_fn,
                persistent_workers=True,
            )

            dataloaders.append(dl)

        return dataloaders

    def _build_audio_text_datasets(self, cfg):
        datasets = []

        for meta_path in cfg.meta_paths:
            df = pl.read_parquet(meta_path)

            audio_file_paths = df[META_FILEPATHS_KEY].to_list()
            prompts = df[META_PROMPT_KEY].to_list()
            responses = df[META_RESPONSE_KEY].to_list()
            ids = df[META_ID_KEY].to_list()

            audio_dataset = AudioFeatureDataset(
                cfg=cfg.audio_cfg,
                file_paths=audio_file_paths,
            )
            text_dataset = QATextDataset(
                prompts=prompts,
                responses=responses,
                tokenizer=self.tokenizer,
                prompt_template_filepath=cfg.text_cfg.prompt_template_filepath,
                text_preprocessor=cfg.text_cfg.text_preprocessor,
                response_prefix=cfg.text_cfg.response_prefix,
                prune_tokens_map_path=cfg.text_cfg.prune_tokens_map_path,
            )
            audio_text_dataset = AudioQATextDataset(
                audio_dataset=audio_dataset,
                text_dataset=text_dataset,
                ids=ids,
            )

            datasets.append(audio_text_dataset)

        return datasets

    def _build_concat_dataset(self, cfg, weighted=False):
        if weighted:
            dataset = WeightedConcatDataset(
                datasets=self._build_audio_text_datasets(cfg),
                dataset_weights=cfg.dataset_weights,
                samples_per_epoch=cfg.samples_per_epoch,
            )
        else:
            dataset = ConcatDataset(datasets=self._build_audio_text_datasets(cfg))

        return dataset

    def _build_test_datasets(self):
        datasets = []

        for k, cfg in enumerate(self.test_cfgs):
            test_dataset = ALMEvalDataset(
                dataset=self._build_concat_dataset(cfg, weighted=False),
                evaluator=self.evaluators[k],
            )

            datasets.append(test_dataset)

        return datasets

    def _custom_collate_fn(
        self,
        batch: list[AudioQATextData],
    ) -> dict[str, any]:
        audio_features_list = []
        audio_to_sample_assignment = []
        audio_order_in_sample = []
        audio_wav_list = []
        pre_audio_tokens_batch = []
        post_audio_tokens_batch = []
        response_mask_batch = []
        unique_id_list = []

        features_key = get_features_key(
            batch[0].audio_feature_data.audio_features.keys()
        )

        for i, item in enumerate(batch):
            audio_features_list.append(item.audio_feature_data.audio_features)
            num_audios_in_item = item.audio_feature_data.audio_features[
                features_key
            ].shape[0]
            audio_to_sample_assignment.append(torch.ones(num_audios_in_item) * i)
            audio_order_in_sample.append(torch.arange(num_audios_in_item))

            audio_wav_list.append(item.audio_feature_data.audio_wav)
            pre_audio_tokens_batch.append(
                {"input_ids": item.qa_text_data.pre_audio_tokens}
            )
            post_audio_tokens_batch.append(
                {"input_ids": item.qa_text_data.post_audio_tokens}
            )
            response_mask_batch.append(item.qa_text_data.response_mask)
            unique_id_list.append(item.unique_id)

        audio_features_batch = {}
        for key in audio_features_list[0].keys():
            tensors_list = [features[key] for features in audio_features_list]
            audio_features_batch[key] = torch.concatenate(tensors_list, dim=0)

        metadata_batch = {}
        audio_wav_batch = torch.concatenate(audio_wav_list, dim=0)
        metadata_batch[AUDIO_WAV_KEY] = audio_wav_batch
        audio_to_sample_assignment = torch.concatenate(audio_to_sample_assignment)
        metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY] = audio_to_sample_assignment
        audio_order_in_sample = torch.concatenate(audio_order_in_sample)
        metadata_batch[AUDIO_ORDER_IN_SAMPLE_KEY] = audio_order_in_sample
        metadata_batch[META_ID_KEY] = unique_id_list

        pre_audio_tokens_batch = self.tokenizer.pad(
            pre_audio_tokens_batch, padding=True, return_tensors="pt"
        )
        post_audio_tokens_batch = self.tokenizer.pad(
            post_audio_tokens_batch, padding=True, return_tensors="pt"
        )
        response_mask_batch = pad_sequence(
            [torch.tensor(rm) for rm in response_mask_batch],
            batch_first=True,
            padding_value=0,
        )

        return {
            "audio_features_batch": audio_features_batch,
            "metadata_batch": metadata_batch,
            "pre_audio_tokens_batch": pre_audio_tokens_batch,
            "post_audio_tokens_batch": post_audio_tokens_batch,
            "response_mask_batch": response_mask_batch,
        }
