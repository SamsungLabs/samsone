import random

import torch.nn as nn


class Compose(nn.Module):
    def __init__(self, transforms: list[nn.Module], shuffle: bool = False):
        super().__init__()
        self.transforms = transforms
        self.shuffle = shuffle

    def forward(self, data, attention_mask):
        if self.shuffle:
            random.shuffle(self.transforms)

        for transform in self.transforms:
            data, attention_mask = transform(data, attention_mask)

        return data, attention_mask


class OneOf(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, data, attention_mask):
        transform = random.choice(self.transforms)

        return transform(data, attention_mask)
