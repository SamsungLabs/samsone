import torch
import torch.nn as nn
import torch.nn.functional as F


class ToMono(nn.Module):
    """
    Converts a multi-channel audio waveform to mono by averaging the channels.
    Input shape: [channels, time]
    Output shape: [1, time]
    """

    def forward(
        self, waveform: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the input waveform to mono.

        Args:
            waveform (torch.Tensor): The input audio waveform tensor,
                                     expected shape [channels, time].

        Returns:
            torch.Tensor: The mono audio waveform tensor, shape [1, time].
        """
        if waveform.ndim == 1:
            return waveform.unsqueeze(0), attention_mask.unsqueeze(0)
        if waveform.shape[0] == 1:
            return waveform, attention_mask

        return waveform.mean(dim=0, keepdim=True), attention_mask[0:1]


class CropOrPad(nn.Module):
    """
    Randomly crops or pads audio to a desired segment length.
    Input shape: [channels, time]
    Output shape: [channels, segment_length]
    """

    def __init__(self, segment_length: int):
        super().__init__()
        self.segment_length = segment_length

    def forward(
        self, waveform: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diff = waveform.shape[-1] - self.segment_length

        if diff > 0:
            # Crop
            output = waveform[:, : self.segment_length]
            new_attention_mask = attention_mask[:, : self.segment_length]
        elif diff < 0:
            # Pad
            pad_len = -diff
            pad = (0, pad_len)
            output = F.pad(waveform, pad, value=0.0)
            new_attention_mask = F.pad(attention_mask, pad, value=0)
        else:
            # Length is already correct
            output = waveform
            new_attention_mask = attention_mask

        return output, new_attention_mask
