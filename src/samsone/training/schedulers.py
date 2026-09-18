import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for the linear warmup phase.
            max_steps (int): Total number of steps (warmup + cosine annealing).
            min_lr (float): Minimum learning rate (end of cosine, start of warmup).
            last_epoch (int): The index of the last epoch/step. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1. Linear Warmup Phase
        if self.last_epoch < self.warmup_steps:
            # Linear increase from min_lr to base_lr
            alpha = self.last_epoch / self.warmup_steps
            return [
                self.min_lr + (base_lr - self.min_lr) * alpha
                for base_lr in self.base_lrs
            ]

        # 2. Cosine Annealing Phase
        else:
            # Calculate progress within the cosine phase (0.0 to 1.0)
            progress = (self.last_epoch - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)  # Clip to ensure we don't go past end

            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
