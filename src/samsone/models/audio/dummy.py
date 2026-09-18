import torch
import torch.nn as nn

from samsone.models.audio.poolers import DefaultAudioEmbeddingsPooler
from samsone.models.base import Model


class DummyAudioModel(Model):
    """
    A simple dummy model for processing audio data.
    It applies a linear layer to each time step and then aggregates the time dimension.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_size: int,
        pooler: DefaultAudioEmbeddingsPooler = DefaultAudioEmbeddingsPooler(),
    ):
        """
        Args:
            feature_dim (int): The number of features in the input spectrogram.
            embedding_size (int): The size of the output embedding after aggregation.
        """
        super().__init__()
        self.linear = nn.Linear(feature_dim, embedding_size)
        self.pooler = pooler

    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x (dict): Input dict containing tensor of shape (batch_size, num_audios, time_dim, feature_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_audios, time_dim, embedding_size).
        """
        x = x["input_features"]
        # Input shape: (batch_size, num_audios, time_dim, feature_dim)
        # Output shape: (batch_size, num_audios, time_dim, embedding_size)
        x = self.linear(x)
        x = self.pooler(x)

        return x
