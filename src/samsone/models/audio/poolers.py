from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AudioEmbeddingsPooler(nn.Module, ABC):
    """
    Abstract base class for audio embeddings poolers.
    Processes audio embeddings (last_hidden_state) and returns audio tokens for use in audio language models.
    """

    @abstractmethod
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process audio embeddings and return audio tokens.

        Args:
            last_hidden_state: Audio embeddings
            attention_mask: Attention mask

        Returns:
            torch.Tensor: Output tokens of shape [batch_size, sequence_length, embedding_dim]
            torch.Tensor: Attention mask of shape [batch_size, sequence_length]
        """
        pass


class DefaultAudioEmbeddingsPooler(AudioEmbeddingsPooler):
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = last_hidden_state.mean(dim=1, keepdim=True)
        attention_mask = torch.ones(output.shape[:2], device=output.device)

        return output, attention_mask


class CLAPAudioEmbeddingsPooler(AudioEmbeddingsPooler):
    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # last_hidden_state.shape = [batch_size, n_channels, freq_shape, temporal_shape]
        output = last_hidden_state.permute((0, 2, 3, 1)).mean(dim=1)
        attention_mask = torch.ones(output.shape[:2], device=output.device)

        return output, attention_mask


class FixedLengthAudioEmbeddingsPooler(AudioEmbeddingsPooler):
    def __init__(self, num_output_tokens: int = 50):
        super().__init__()
        self.num_output_tokens = num_output_tokens

    @staticmethod
    def pool_to_fixed_len(
        x: torch.Tensor, output_seq_len: int, pool_mode: str
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        remaining_dims = x.shape[2:]

        pool_size = max(1, seq_len // output_seq_len)
        actual_output_tokens = seq_len // pool_size
        trimmed_len = actual_output_tokens * pool_size
        x = x[:, :trimmed_len, ...]

        new_shape = (batch_size, actual_output_tokens, pool_size) + remaining_dims

        reshaped_x = x.view(new_shape)

        if pool_mode == "avg":
            pooled_x = reshaped_x.mean(dim=2)
        elif pool_mode == "max":
            pooled_x, _ = reshaped_x.max(dim=2)
        else:
            raise NotImplementedError("Pool mode has to be either 'avg' or 'max'!")

        return pooled_x

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_features = FixedLengthAudioEmbeddingsPooler.pool_to_fixed_len(
            last_hidden_state, self.num_output_tokens, pool_mode="avg"
        )
        pooled_attention_mask = FixedLengthAudioEmbeddingsPooler.pool_to_fixed_len(
            attention_mask, self.num_output_tokens, pool_mode="max"
        )

        return pooled_features, pooled_attention_mask


class RatioAudioEmbeddingsPooler(AudioEmbeddingsPooler):
    def __init__(self, downsample_ratio: int = 8):
        super().__init__()
        self.downsample_ratio = downsample_ratio

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # last_hidden_state shape: [batch_size, seq_len, embedding_dim]: [batch_size, 1500, 384]
        batch_size, seq_len, embedding_dim = last_hidden_state.shape

        to_trim = seq_len % self.downsample_ratio

        # return original sequence if shorter than downsample ratio
        if to_trim == seq_len:
            return last_hidden_state, attention_mask

        trimmed_hidden_state = last_hidden_state[:, :-to_trim, :]
        trimmed_attn_mask = attention_mask[:, :-to_trim]
        seq_len = trimmed_hidden_state.shape[1]

        # Reshape for pooling: [batch, actual_output_tokens, pool_size, embedding_dim]
        reshaped_features = trimmed_hidden_state.view(
            batch_size,
            seq_len // self.downsample_ratio,
            self.downsample_ratio,
            embedding_dim,
        )
        reshaped_attention_mask = trimmed_attn_mask.view(
            batch_size,
            seq_len // self.downsample_ratio,
            self.downsample_ratio,
        )
        pooled_features = reshaped_features.mean(
            dim=2
        )  # [batch, actual_output_tokens, embedding_dim]
        pooled_attention_mask, _ = reshaped_attention_mask.max(
            dim=2
        )  # [batch, actual_output_tokens]

        return pooled_features, pooled_attention_mask


class ASTAudioEmbeddingsPooler(AudioEmbeddingsPooler):
    def __init__(
        self, num_output_tokens: int = 50, include_cls_and_dist_embs: bool = True
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.include_cls_and_dist_embs = include_cls_and_dist_embs

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # last_hidden_state: [batch, 1214, 768]
        batch_size, seq_len, embedding_dim = last_hidden_state.shape

        # Remove CLS and distillation tokens (first 2 tokens)
        patch_embeddings = last_hidden_state[:, 2:, :]  # [batch, 1212, 768]

        if self.include_cls_and_dist_embs:
            cls_and_dist_embs = last_hidden_state[:, :2, :]
            combined_embs = cls_and_dist_embs.mean(dim=1, keepdim=True)
            patch_embeddings += combined_embs

        # Reshape to 2D grid: [batch, 12, 101, 768]
        frequency_patches = 12
        time_patches = 101
        reshaped = patch_embeddings.view(
            batch_size, frequency_patches, time_patches, embedding_dim
        )

        # Pool along frequency dimension
        frequency_pooled = reshaped.mean(dim=1)  # [batch, 101, 768]

        # Time-based pooling to desired number of tokens
        current_time_tokens = frequency_pooled.shape[1]  # 101
        pool_size = max(
            1, current_time_tokens // self.num_output_tokens
        )  # 101 // num_output_tokens
        actual_output_tokens = current_time_tokens // pool_size

        # Trim to make divisible by pool_size
        trimmed_len = actual_output_tokens * pool_size
        trimmed_embeddings = frequency_pooled[
            :, :trimmed_len, :
        ]  # [batch, actual_output_tokens * pool_size, 768]

        # Reshape for time pooling: [batch, actual_output_tokens, pool_size, 768]
        time_reshaped = trimmed_embeddings.view(
            batch_size, actual_output_tokens, pool_size, embedding_dim
        )
        time_pooled = time_reshaped.mean(dim=2)  # [batch, actual_output_tokens, 768]
        attention_mask = attention_mask[:, :actual_output_tokens]

        return time_pooled, attention_mask


_HUGGINGFACE_POOLERS = {
    "laion/clap-htsat-unfused": CLAPAudioEmbeddingsPooler,
    "openai/whisper-tiny": FixedLengthAudioEmbeddingsPooler,
    "openai/whisper-small": FixedLengthAudioEmbeddingsPooler,
    "ntu-spml/distilhubert": RatioAudioEmbeddingsPooler,
    "MIT/ast-finetuned-audioset-10-10-0.4593": ASTAudioEmbeddingsPooler,
}


class HuggingFaceAudioEmbeddingsPooler(AudioEmbeddingsPooler):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        if model_name_or_path not in _HUGGINGFACE_POOLERS:
            print(
                f"No pooler for {model_name_or_path}. The default pooler will be used."
            )

        self.pooler = _HUGGINGFACE_POOLERS.get(
            model_name_or_path, DefaultAudioEmbeddingsPooler
        )()

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pooler(last_hidden_state, attention_mask)
