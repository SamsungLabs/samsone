from abc import ABC, abstractmethod

import torch
from transformers import AutoModel, PreTrainedModel

from samsone.models.audio.poolers import AudioEmbeddingsPooler
from samsone.models.base import Model


class AudioEncoder(Model, ABC):
    """
    Abstract base class for audio encoders.
    Processes audio features and returns embeddings for use in audio language models.
    """

    @abstractmethod
    def forward(self, audio_features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process audio features and return embeddings.

        Args:
            audio_features: Dict containing multiple feature tensors

        Returns:
            torch.Tensor: Output embeddings of shape [batch_size, sequence_length, embedding_dim]
            torch.Tensor: Embedding attention mask of shape [batch_size, sequence_length]
        """
        pass


class HuggingFaceAudioEncoder(AudioEncoder):
    """
    Audio encoder using HuggingFace pre-trained audio models.
    """

    def __init__(self, model_name_or_path: str):
        """
        Args:
            model_name_or_path (str): Name or path of the HuggingFace audio model to use.
        """
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, audio_features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process audio features using the HuggingFace audio model.

        Args:
            audio_features: Dict containing multiple feature tensors (e.g., input_values, attention_mask)

        Returns:
            torch.Tensor: Output embeddings of shape [batch_size, sequence_length, embedding_dim]
            torch.Tensor: Embedding attention mask of shape [batch_size, sequence_length]
        """

        pass


class WhisperAudioEncoder(HuggingFaceAudioEncoder):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.encoder = self.encoder._modules["encoder"]

    def forward(self, audio_features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs = self.encoder(**audio_features).last_hidden_state
        # the model doesnt support attention masking, therefore all output tokens are valid
        attention_mask = torch.ones(
            encoder_outputs.shape[:2], device=encoder_outputs.device
        )

        return encoder_outputs, attention_mask


class ASTAudioEncoder(HuggingFaceAudioEncoder):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)

    def forward(self, audio_features):
        # remove attention mask, as encoder doesn't expect it
        audio_features.pop("attention_mask", None)
        encoder_outputs = self.encoder(**audio_features).last_hidden_state
        # the model doesnt support attention masking, therefore all output tokens are valid
        attention_mask = torch.ones(
            encoder_outputs.shape[:2], device=encoder_outputs.device
        )

        return encoder_outputs, attention_mask


class HubertAudioEncoder(HuggingFaceAudioEncoder):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)

    def forward(self, audio_features):
        encoder_outputs = self.encoder(**audio_features).last_hidden_state

        # create attention mask
        input_lengths = audio_features["attention_mask"].sum(dim=-1)
        valid_output_lengths = self.encoder._get_feat_extract_output_lengths(
            input_lengths
        )  # get num valid features based on num valid samples
        seq_range = torch.arange(
            encoder_outputs.shape[1], device=encoder_outputs.device
        ).expand(encoder_outputs.shape[0], encoder_outputs.shape[1])
        attention_mask = (seq_range < valid_output_lengths.unsqueeze(-1)).int()

        return encoder_outputs, attention_mask


class AudioEncoderWithPooling(AudioEncoder):
    def __init__(self, encoder: AudioEncoder, pooler: AudioEmbeddingsPooler):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler

    def forward(self, audio_features: dict) -> torch.Tensor:
        """
        Process audio features using the AudioEncoder and pool them to produce audio tokens embeddings.

        Args:
            audio_features: Dict containing multiple feature tensors (e.g., input_values, attention_mask)

        Returns:
            torch.Tensor: Output audio embeddings after pooling
        """

        encoder_outputs, attention_mask = self.encoder(audio_features)
        audio_embeddings, pooled_attention_mask = self.pooler(
            encoder_outputs, attention_mask
        )

        return audio_embeddings, pooled_attention_mask


if __name__ == "__main__":
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    encoder = HuggingFaceAudioEncoder(model_name)
    print("✓ Encoder initialized successfully!")

    # Create random input tensor: [batch_size, num_audios, channels, sequence_length, features]
    batch_size, num_audios, channels, sequence_length, features = 2, 3, 1, 1024, 128
    random_audio_features = torch.randn(
        batch_size, num_audios, channels, sequence_length, features
    )

    print(f"Input tensor shape: {random_audio_features.shape}")

    with torch.no_grad():
        output_embeddings = encoder(random_audio_features)

    print(f"Output embeddings shape: {output_embeddings.shape}")
    print("✓ Forward pass completed successfully!")
