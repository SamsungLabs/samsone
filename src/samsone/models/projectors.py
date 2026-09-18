from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from samsone.models.base import Model


class Projector(Model, ABC):
    """
    Abstract base class for projectors.
    A projector transforms a sequence of embeddings from one shape to another.
    Input shape: (batch_size, seq_len, input_embedding_dim)
    Output shape: (batch_size, seq_len, output_embedding_dim)
    """

    def __init__(self, input_embedding_dim: int, output_embedding_dim: int):
        super().__init__()
        self.input_embedding_dim = input_embedding_dim
        self.output_embedding_dim = output_embedding_dim

    @abstractmethod
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, input_embedding_dim)
        # attention_mask: (batch_size, seq_len)
        # Expected output: (batch_size, seq_len, output_embedding_dim), attention_mask: (batch_size, seq_len)
        pass


class LinearProjector(Projector):
    """
    A simple projector that uses a single linear layer to project embeddings.
    """

    def __init__(self, input_embedding_dim: int, output_embedding_dim: int):
        super().__init__(input_embedding_dim, output_embedding_dim)
        self.linear = nn.Linear(self.input_embedding_dim, self.output_embedding_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, input_embedding_dim)
        x = self.linear(x)
        # x: (batch_size, seq_len, output_embedding_dim)
        return x, attention_mask


class NonLinearProjector(Projector):
    """
    A nonlinear projector that uses two linear layers with residual connection,
    GELU activation, dropout, and layer normalization.
    """

    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        dropout_p: float = 0.5,
    ):
        super().__init__(input_embedding_dim, output_embedding_dim)
        self.linear1 = nn.Linear(
            self.input_embedding_dim, self.output_embedding_dim, bias=False
        )
        self.linear2 = nn.Linear(
            self.output_embedding_dim, self.output_embedding_dim, bias=False
        )
        self.layer_norm = nn.LayerNorm(self.output_embedding_dim)
        self.dropout = nn.Dropout(dropout_p)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, input_embedding_dim)
        embed1 = self.linear1(x)
        embed2 = self.dropout(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        # embeds: (batch_size, seq_len, output_embedding_dim)
        return embeds, attention_mask
