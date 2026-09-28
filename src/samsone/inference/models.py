"""Config-free, ready-to-run SAMSONE inference models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TypeAlias

import torch
from torchcodec.decoders import AudioDecoder
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from samsone.data.constants import ATTENTION_MASK_KEY, AUDIO_TO_SAMPLE_ASSIGNMENT_KEY
from samsone.data.preprocessing.audio import CropOrPad, ToMono, WhisperFeatureExtractor
from samsone.models.audio import (
    AudioEncoderWithPooling,
    HuggingFaceAudioEmbeddingsPooler,
    WhisperAudioEncoder,
)
from samsone.models.multimodal import AudioLM
from samsone.models.projectors import NonLinearProjector
from samsone.models.text.lm import GenerationConfig, HuggingFaceLM

from .download import CACHE_DIR, get_checkpoint_path

AudioInput: TypeAlias = str | Path | Sequence[str | Path]


@dataclass(frozen=True)
class _ModelSpec:
    checkpoint_filename: str
    text_model_name: str
    output_embedding_dim: int
    num_hidden_layers: int | None = None


def _build_pruned_token_map(tokenizer: PreTrainedTokenizerBase) -> dict[int, int]:
    """Recreate the token map used by the released SAMSONE checkpoints."""
    special_token_ids = set(tokenizer.all_special_ids)
    token_map: dict[int, int] = {}
    for token, original_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: item[1]
    ):
        cleaned = re.sub(r"[^A-Za-z0-9]", "", token)
        should_keep = original_id in special_token_ids or (
            not any(character.isupper() for character in cleaned)
            and not re.search(r"[\sĠĊĉ]{4,}", token)
            and not re.search(r"[^\x00-\x7FĠĊĉ\s]", token)
            and not re.search(r"[-#]{3,}", token)
        )
        if should_keep:
            token_map[len(token_map)] = original_id
    return token_map


class SamsoneModel:
    """A released SAMSONE audio-language model.

    A list of audio paths is interpreted as multiple clips for one prompt and
    produces one generated response.
    """

    spec: _ModelSpec
    sample_rate = 16_000
    max_audio_samples = 480_000

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_path: str | Path | None = None,
        checkpoint_url: str | None = None,
        max_new_tokens: int = 100,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        path = (
            Path(checkpoint_path)
            if checkpoint_path
            else get_checkpoint_path(self.spec.checkpoint_filename, url=checkpoint_url)
        )
        self.model = self._build_model()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        if not all(key.startswith("audio_lm.") for key in state_dict):
            raise ValueError(
                "Expected an AudioLMExperiment checkpoint state dictionary."
            )
        model_state_dict = {
            key.removeprefix("audio_lm."): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.feature_extractor = WhisperFeatureExtractor(
            "openai/whisper-tiny", sample_rate=self.sample_rate
        )
        self.to_mono = ToMono()
        self.crop_or_pad = CropOrPad(self.max_audio_samples)
        self._original_to_pruned_token_id = {
            original_id: pruned_id
            for pruned_id, original_id in self.model.text_model.tokens_map.items()
        }

    def _build_model(self) -> AudioLM:
        tokenizer = AutoTokenizer.from_pretrained(self.spec.text_model_name)
        token_map_path = (
            CACHE_DIR / f"{self.spec.text_model_name.rsplit('/', 1)[-1]}_prune_map.json"
        )
        if not token_map_path.is_file():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            import json

            token_map_path.write_text(json.dumps(_build_pruned_token_map(tokenizer)))
        text_model = HuggingFaceLM(
            model_name_or_path=self.spec.text_model_name,
            generation_config=GenerationConfig(),
            num_hidden_layers=self.spec.num_hidden_layers,
            prune_tokens_map_path=str(token_map_path),
            load_pretrained_weights=False,
        )
        return AudioLM(
            audio_encoder=AudioEncoderWithPooling(
                encoder=WhisperAudioEncoder(
                    "openai/whisper-tiny", load_pretrained_weights=False
                ),
                pooler=HuggingFaceAudioEmbeddingsPooler("openai/whisper-tiny"),
            ),
            projector=NonLinearProjector(
                384, self.spec.output_embedding_dim, dropout_p=0.5
            ),
            text_model=text_model,
        )

    def __call__(
        self, *, audio: AudioInput, prompt: str, max_new_tokens: int | None = None
    ) -> str:
        paths = self._normalise_audio_paths(audio)
        with torch.inference_mode():
            output = self.model.generate(
                audio_features_batch=self._prepare_audio(paths),
                metadata_batch={
                    AUDIO_TO_SAMPLE_ASSIGNMENT_KEY: torch.zeros(
                        len(paths), dtype=torch.long, device=self.device
                    )
                },
                pre_audio_tokens_batch=self._tokenize(""),
                post_audio_tokens_batch=self._tokenize(
                    self._normalise_prompt(prompt) + " answer: "
                ),
                max_new_tokens=max_new_tokens or self.max_new_tokens,
            )
        return output[0].strip()

    @staticmethod
    def _normalise_audio_paths(audio: AudioInput) -> list[Path]:
        paths = (
            [Path(audio)]
            if isinstance(audio, (str, Path))
            else [Path(path) for path in audio]
        )
        if not paths:
            raise ValueError("audio must contain at least one audio file path.")
        missing_paths = [str(path) for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"Audio file(s) not found: {', '.join(missing_paths)}"
            )
        return paths

    @staticmethod
    def _normalise_prompt(prompt: str) -> str:
        prompt = re.sub(r"[^\x00-\x7F]", "", prompt.lower())
        prompt = re.sub(r"\s{4,}", " ", prompt)
        return re.sub(r"#{3,}", "#", re.sub(r"-{3,}", "-", prompt))

    def _prepare_audio(self, paths: list[Path]) -> dict[str, torch.Tensor]:
        features_by_key: dict[str, list[torch.Tensor]] = {}
        for path in paths:
            waveform = (
                AudioDecoder(path, sample_rate=self.sample_rate).get_all_samples().data
            )
            attention_mask = torch.ones_like(waveform)
            waveform, attention_mask = self.to_mono(waveform, attention_mask)
            waveform, attention_mask = self.crop_or_pad(waveform, attention_mask)
            for key, value in self.feature_extractor(waveform, attention_mask).items():
                features_by_key.setdefault(key, []).append(value)
        audio_features = {}
        for key, values in features_by_key.items():
            features = torch.cat(values)
            if features.is_floating_point() and key != ATTENTION_MASK_KEY:
                features = features.to(device=self.device, dtype=self.dtype)
            else:
                features = features.to(self.device)
            audio_features[key] = features
        return audio_features

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        tokens = self.model.text_model.tokenizer(
            [text], return_tensors="pt", padding=True
        )
        input_ids = tokens["input_ids"][0].tolist()
        try:
            mapped_ids = [
                self._original_to_pruned_token_id[token_id] for token_id in input_ids
            ]
        except KeyError as error:
            raise ValueError(
                "Prompt contains a token unavailable in this model's vocabulary."
            ) from error
        return {
            "input_ids": torch.tensor(
                [mapped_ids], dtype=torch.long, device=self.device
            ),
            ATTENTION_MASK_KEY: tokens[ATTENTION_MASK_KEY].to(self.device),
        }


class Samsone99M(SamsoneModel):
    spec = _ModelSpec(
        "samsone_99m_model_bf16.ckpt", "HuggingFaceTB/SmolLM2-135M", 576, 20
    )


class Samsone134M(SamsoneModel):
    spec = _ModelSpec("samsone_134m_model_bf16.ckpt", "HuggingFaceTB/SmolLM2-135M", 576)


class Samsone356M(SamsoneModel):
    spec = _ModelSpec("samsone_356m_model_bf16.ckpt", "HuggingFaceTB/SmolLM2-360M", 960)
