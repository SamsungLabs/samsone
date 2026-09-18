import torch
import torch.nn as nn


class SpecNormalization(nn.Module):
    def __init__(self, dim: int = -2):
        super().__init__()
        self.dim = dim

    def forward(
        self, spec: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = spec - torch.mean(spec, dim=self.dim, keepdim=True)

        return spec, attention_mask
