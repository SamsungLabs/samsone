from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Literal

from filelock import FileLock
import lightning.pytorch as L
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
import numpy as np
import torch

from samsone.data.constants import (
    AUDIO_ORDER_IN_SAMPLE_KEY,
    AUDIO_TO_SAMPLE_ASSIGNMENT_KEY,
    AUDIO_WAV_KEY,
    META_ID_KEY,
)
from samsone.experiments.constants import (
    GENERATED_TEST_ANSWERS_FILENAME,
)
from samsone.models.base import FreezeConfig
from samsone.models.multimodal.audio_lm import AudioLM, AudioLMForExport

logger = logging.getLogger("lightning")


def setup_results_dir(trainer, logger):
    """Setup shared results directory for multi-GPU setups."""
    # Check if we're in distributed mode and has is_global_zero attribute
    is_distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    is_main_process = getattr(
        trainer, "is_global_zero", True
    )  # Default to True for single GPU

    # Main process determines and broadcasts the path
    if is_main_process:
        main_process_path = str(Path(logger.save_dir) / "evaluation_results")
    else:
        main_process_path = None

    # Broadcast path to all processes if in distributed mode
    if is_distributed:
        path_obj = [main_process_path]
        torch.distributed.broadcast_object_list(path_obj, src=0)
        shared_path = path_obj[0]
    else:
        # Single process case - use local path
        shared_path = str(Path(logger.save_dir) / "evaluation_results")

    # Main process creates root directory
    if is_main_process:
        Path(shared_path).mkdir(parents=True, exist_ok=True)

    # Synchronize if in distributed mode
    if is_distributed:
        torch.distributed.barrier()

    return Path(shared_path)


@dataclass
class SummaryConfig:
    print_summary: bool = True
    freeze_summary_depth: int = 1


@dataclass
class LoggingConfig:
    sample_rate: int
    log_generate_train_step: int = 100
    log_generate_val_step: int = 100
    log_generate_test_step: int = 10


class AudioLMExperiment(L.LightningModule):
    def __init__(
        self,
        audio_lm_model: AudioLM | AudioLMForExport,
        freeze_config: FreezeConfig,
        summary_config: SummaryConfig,
        max_train_seq_tokens: int,
        max_new_tokens: int,
        logging_config: LoggingConfig,
        optimizer: OptimizerCallable,
        lr_scheduler: LRSchedulerCallable | None = None,
        model_ckpt_load_path: str | None = None,
        force_generate_test: bool = False,
    ):
        super().__init__()
        self.audio_lm = audio_lm_model

        if model_ckpt_load_path:
            checkpoint = torch.load(model_ckpt_load_path)
            self.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Loaded model from checkpoint {model_ckpt_load_path}")

        self.logging_config = logging_config

        self.audio_lm.train()
        self.audio_lm.freeze(freeze_config.freeze_modules)

        self.max_train_seq_tokens = max_train_seq_tokens
        self.max_new_tokens = max_new_tokens

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.save_hyperparameters(ignore=["audio_lm_model"])

        if summary_config.print_summary:
            self.audio_lm.freeze_summary(depth=summary_config.freeze_summary_depth)

        self.force_generate_test = force_generate_test

    def forward(
        self,
        audio_features_batch: dict,
        metadata_batch: dict,
        pre_audio_tokens_batch: dict,
        post_audio_tokens_batch: dict,
        response_mask_batch: torch.Tensor,
    ):
        # The forward pass of the experiment is the forward pass of the model
        return self.audio_lm(
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_mask_batch,
            max_tokens=self.max_train_seq_tokens,
        )

    def generate(
        self,
        audio_features_batch: dict,
        metadata_batch: dict,
        pre_audio_tokens_batch: list[str],
        post_audio_tokens_batch: list[str],
        max_new_tokens: int,
    ) -> list[str]:
        return self.audio_lm.generate(
            audio_features_batch=audio_features_batch,
            metadata_batch=metadata_batch,
            pre_audio_tokens_batch=pre_audio_tokens_batch,
            post_audio_tokens_batch=post_audio_tokens_batch,
            max_new_tokens=max_new_tokens,
        )

    def _common_step(self, batch, batch_idx, tag: Literal["train", "val"]):
        (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_mask_batch,
        ) = self._unpack_batch(batch)

        model_output, _ = self(  # model_output is a Hugging Face ModelOutput object
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_mask_batch,
        )

        self._log_model_generation(batch, batch_idx, tag)

        # The loss is calculated internally by the Hugging Face model
        # when labels are provided to its forward method.
        loss = model_output.loss

        return loss

    def _prepare_data_for_generation(
        self,
        batch,
    ) -> tuple[dict, dict, dict, dict, torch.Tensor | None]:
        """
        Prepare batch data for generation by removing response tokens and padding.

        Args:
            batch: The batch data (assumed to be already properly filtered/structured)

        Returns:
            Tuple of (audio_features_batch, metadata_batch, pre_audio_tokens_batch,
                     post_audio_tokens_batch, response_ids)
        """
        (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_mask_batch,
        ) = self._unpack_batch(batch)

        batch_size = post_audio_tokens_batch["input_ids"].shape[0]
        response_ids = [
            torch.masked_select(
                post_audio_tokens_batch["input_ids"][i : i + 1],
                response_mask_batch[i].bool(),
            )
            for i in range(batch_size)
        ]

        filtered_post_audio_tokens_batch = defaultdict(list)
        batch_size = post_audio_tokens_batch["input_ids"].shape[0]

        for i in range(batch_size):
            sample_response_mask = response_mask_batch[i]

            for key, value in post_audio_tokens_batch.items():
                # Remove response tokens and keep only prompt tokens
                filtered_value = torch.masked_select(
                    value[i : i + 1],
                    ~sample_response_mask.bool(),
                )
                filtered_post_audio_tokens_batch[key].append(filtered_value)

            valid_token_mask = filtered_post_audio_tokens_batch["attention_mask"][
                i
            ].bool()
            filtered_post_audio_tokens_batch["input_ids"][i] = torch.masked_select(
                filtered_post_audio_tokens_batch["input_ids"][i],
                valid_token_mask,
            )
            filtered_post_audio_tokens_batch["attention_mask"][i] = torch.masked_select(
                filtered_post_audio_tokens_batch["attention_mask"][i],
                valid_token_mask,
            )

        filtered_post_audio_tokens_batch = self.audio_lm.text_model.tokenizer.pad(
            filtered_post_audio_tokens_batch, padding=True, return_tensors="pt"
        ).to(post_audio_tokens_batch["input_ids"].device)

        return (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            filtered_post_audio_tokens_batch,
            response_ids,
        )

    def _extract_single_sample(self, batch, sample_index: int) -> dict:
        """
        Extract a single sample from a batch for logging.

        Args:
            batch: The original batch data
            sample_index: Index of the sample to extract

        Returns:
            Filtered batch containing only the specified sample
        """
        (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_mask_batch,
        ) = self._unpack_batch(batch)

        sample_indices = metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY]
        sample_audio_mask = sample_indices == sample_index

        # Filter audio features for the selected sample
        filtered_audio_features_batch = {
            k: v[sample_audio_mask, ...] for k, v in audio_features_batch.items()
        }

        # Filter metadata for the selected sample
        filtered_metadata_batch = {
            k: v[sample_audio_mask, ...]
            if isinstance(v, torch.Tensor)
            else v[sample_index]
            for k, v in metadata_batch.items()
        }
        filtered_metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY] = (
            filtered_metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY] * 0
        )  # Sample index becomes 0

        # Filter pre_audio_tokens for the selected sample
        filtered_pre_audio_tokens_batch = {
            k: v[sample_index : sample_index + 1]
            for k, v in pre_audio_tokens_batch.items()
        }

        # Filter post_audio_tokens for the selected sample
        filtered_post_audio_tokens_batch = {
            k: v[sample_index : sample_index + 1]
            for k, v in post_audio_tokens_batch.items()
        }

        filtered_response_mask_batch = response_mask_batch[
            sample_index : sample_index + 1
        ]

        return {
            "audio_features_batch": filtered_audio_features_batch,
            "metadata_batch": filtered_metadata_batch,
            "pre_audio_tokens_batch": filtered_pre_audio_tokens_batch,
            "post_audio_tokens_batch": filtered_post_audio_tokens_batch,
            "response_mask_batch": filtered_response_mask_batch,
        }

    def _log_model_generation(
        self, batch, batch_idx, tag: Literal["train", "val", "test"]
    ):
        match tag:
            case "train":
                log_step = self.logging_config.log_generate_train_step
                step_num = self.global_step
            case "val":
                log_step = self.logging_config.log_generate_val_step
                step_num = self.global_step + self.val_log_counter
            case "test":
                log_step = self.logging_config.log_generate_test_step
                step_num = self.global_step + self.test_log_counter

        if batch_idx % log_step != 0:
            return

        match tag:
            case "val":
                self.val_log_counter += 1
            case "test":
                self.test_log_counter += 1

        batch_size = batch["post_audio_tokens_batch"]["input_ids"].shape[0]
        selected_batch_idx = np.random.randint(batch_size)

        # Extract single sample for logging
        single_sample_batch = self._extract_single_sample(batch, selected_batch_idx)

        # Prepare data for generation with response tokens for comparison
        (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_ids,
        ) = self._prepare_data_for_generation(single_sample_batch)
        with torch.no_grad():
            generation_output = self.audio_lm.generate(
                audio_features_batch,
                metadata_batch,
                pre_audio_tokens_batch,
                post_audio_tokens_batch,
                max_new_tokens=self.max_new_tokens,
            )[0]

        input_decoded = self.audio_lm.text_model.decode(
            post_audio_tokens_batch["input_ids"][0]
        )
        ground_truth_decoded = self.audio_lm.text_model.decode(response_ids[0])

        text_log = (
            f"Prompt: {input_decoded}\n________________________\n"
            f"Ground Truth Caption: {ground_truth_decoded}\n________________________\n"
            f"Generated Caption: {generation_output}"
        )

        self.logger.experiment.add_text(
            f"{tag}/caption", text_log, global_step=step_num
        )
        for i, audio_to_log in enumerate(metadata_batch[AUDIO_WAV_KEY]):
            self.logger.experiment.add_audio(
                f"{tag}/audio_{i}",
                audio_to_log,
                global_step=step_num,
                sample_rate=self.logging_config.sample_rate,
            )

    def training_step(self, batch, batch_idx):
        batch_size = batch["post_audio_tokens_batch"]["input_ids"].shape[0]
        loss = self._common_step(batch, batch_idx, tag="train")
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["post_audio_tokens_batch"]["input_ids"].shape[0]
        loss = self._common_step(batch, batch_idx, tag="val")
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        filtered_batch = self._filter_test_batch(batch, dataloader_idx=dataloader_idx)

        batch_ids = filtered_batch["metadata_batch"][META_ID_KEY]

        if not batch_ids:
            return

        self._log_model_generation(batch, batch_idx, tag="test")

        (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            _,
        ) = self._prepare_data_for_generation(filtered_batch)

        with torch.no_grad():
            generated_text = self.generate(
                audio_features_batch=audio_features_batch,
                metadata_batch=metadata_batch,
                pre_audio_tokens_batch=pre_audio_tokens_batch,
                post_audio_tokens_batch=post_audio_tokens_batch,
                max_new_tokens=self.max_new_tokens,
            )

        for curr_id, model_output in zip(batch_ids, generated_text):
            self.generated_answers[dataloader_idx][curr_id] = model_output

        if batch_idx > 0 and batch_idx % 50 == 0:
            self._save_answers_to_file(dataloader_idx=dataloader_idx)

    def on_validation_epoch_start(self):
        self.val_log_counter = 0

    def on_test_epoch_start(self):
        self.test_log_counter = 0

    def on_test_start(self):
        self.test_results_root_dir_path = setup_results_dir(self.trainer, self.logger)

        self.evaluators = []
        self.generated_answers = []
        self.answer_files = []
        self.result_dir_paths = []

        for idx in range(len(self.trainer.test_dataloaders)):
            evaluator = self.trainer.test_dataloaders[idx].dataset.evaluator
            self.evaluators.append(evaluator)
            self.result_dir_paths.append(
                self.test_results_root_dir_path
                / f"train_epoch={self.last_training_epoch}"
                / evaluator.test_dataset_name
            )

            is_main_process = getattr(self.trainer, "is_global_zero", True)
            if is_main_process:
                self.result_dir_paths[idx].mkdir(parents=True, exist_ok=True)

            self.generated_answers.append({})
            self.answer_files.append(
                self.result_dir_paths[idx] / GENERATED_TEST_ANSWERS_FILENAME
            )

            if not self.force_generate_test:
                if self.answer_files[idx].is_file():
                    with open(self.answer_files[idx], "r") as f:
                        self.generated_answers[idx] = json.load(f)
                    logger.info(
                        f"Loaded {len(self.generated_answers[idx])} saved answers for {evaluator.test_dataset_name}."
                    )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_test_end(self):
        for idx in range(len(self.trainer.test_dataloaders)):
            self._save_answers_to_file(dataloader_idx=idx)

        # Synchronize all processes before evaluation
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if hasattr(self.trainer, "is_global_zero"):
            is_main_process = self.trainer.is_global_zero
        else:
            # Fallback for single GPU or when is_global_zero is not available
            is_main_process = True

        if not is_main_process:
            return

        # Load all answers from files for final evaluation
        for idx in range(len(self.trainer.test_dataloaders)):
            with open(self.answer_files[idx], "r") as f:
                self.generated_answers[idx] = json.load(f)

        # Run evaluators only on main process with complete data
        for idx in range(len(self.trainer.test_dataloaders)):
            self.evaluators[idx].evaluate(self.generated_answers[idx])
            self.evaluators[idx].finalize(results_dir=str(self.result_dir_paths[idx]))

    def on_load_checkpoint(self, checkpoint):
        # extract epoch number for test directory naming
        self.last_training_epoch = checkpoint["epoch"]

    def on_train_end(self):
        # extract epoch number for test directory naming
        self.last_training_epoch = self.trainer.current_epoch - 1

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if not self.lr_scheduler:
            return optimizer

        scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _unpack_batch(self, batch):
        return (
            batch["audio_features_batch"],
            batch["metadata_batch"],
            batch["pre_audio_tokens_batch"],
            batch["post_audio_tokens_batch"],
            batch["response_mask_batch"],
        )

    def _filter_test_batch(self, batch, dataloader_idx):
        (
            audio_features_batch,
            metadata_batch,
            pre_audio_tokens_batch,
            post_audio_tokens_batch,
            response_mask_batch,
        ) = self._unpack_batch(batch)

        processed_samples = []

        for k, sample_id in enumerate(metadata_batch[META_ID_KEY]):
            if sample_id in self.generated_answers[dataloader_idx]:
                logger.debug(f"ID {sample_id} already processed")
                processed_samples.append(k)

        # filter batch if answer was already generated and saved
        for k in reversed(processed_samples):
            idxs_to_remove = torch.where(
                metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY] == k
            )[0]
            last_idx_to_remove = idxs_to_remove[-1]

            # update indices before removing data
            if last_idx_to_remove < (
                metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY].shape[0] - 1
            ):
                metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY][
                    last_idx_to_remove + 1 :, ...
                ] -= 1

            for input_key in audio_features_batch.keys():
                mask = torch.ones(
                    audio_features_batch[input_key].shape[0], dtype=torch.bool
                )
                mask[idxs_to_remove] = False

                audio_features_batch[input_key] = audio_features_batch[input_key][mask]

            for key in [
                AUDIO_TO_SAMPLE_ASSIGNMENT_KEY,
                AUDIO_ORDER_IN_SAMPLE_KEY,
                AUDIO_WAV_KEY,
            ]:
                metadata_batch[key] = metadata_batch[key][mask]

            # remove elements in data that have the same number of elements as batch_size
            bs_mask = torch.ones(len(metadata_batch[META_ID_KEY]), dtype=torch.bool)
            bs_mask[k] = False

            for key in pre_audio_tokens_batch.keys():
                pre_audio_tokens_batch[key] = pre_audio_tokens_batch[key][bs_mask]
                post_audio_tokens_batch[key] = post_audio_tokens_batch[key][bs_mask]

            # Filter response_mask_batch along with other token batches
            if response_mask_batch is not None:
                response_mask_batch = response_mask_batch[bs_mask]

            metadata_batch[META_ID_KEY].pop(k)

        return {
            "audio_features_batch": audio_features_batch,
            "metadata_batch": metadata_batch,
            "pre_audio_tokens_batch": pre_audio_tokens_batch,
            "post_audio_tokens_batch": post_audio_tokens_batch,
            "response_mask_batch": response_mask_batch,
        }

    def _save_answers_to_file(self, dataloader_idx):
        try:
            # Use file locking to prevent race conditions in multi-GPU setup
            lock_file = str(self.answer_files[dataloader_idx]) + ".lock"
            with FileLock(lock_file):
                # Load existing data if file exists
                existing_data = {}
                if self.answer_files[dataloader_idx].is_file():
                    try:
                        with open(self.answer_files[dataloader_idx], "r") as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass

                # Merge existing data with new data (new data takes precedence)
                merged_data = {
                    **existing_data,
                    **self.generated_answers[dataloader_idx],
                }

                # Write merged data back to file
                with open(self.answer_files[dataloader_idx], "w") as f:
                    json.dump(merged_data, f, indent=2)

                # Clear generated answers after saving (except for main process final save)
                self.generated_answers[dataloader_idx].clear()

            logger.debug(
                f"Saved {len(merged_data)} answers to file (was {len(existing_data)} before)"
            )
        except Exception as e:
            logger.error(f"Failed to save answers to file: {e}")
