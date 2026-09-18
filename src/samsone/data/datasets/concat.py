from itertools import cycle
import logging

import numpy as np
from torch.utils.data import Dataset, get_worker_info

logger = logging.getLogger("lightning")


class WeightedConcatDataset(Dataset):
    """
    A PyTorch Dataset that enables weighted sampling from multiple datasets.
    Each dataset is sampled based on specified weights, allowing more frequent
    sampling from certain datasets.

    Args:
        datasets (list of Dataset): List of PyTorch Dataset objects to sample from.
        samples_per_epoch (int): Total number of samples to draw per epoch.
        dataset_weights (list of float, optional): List of weights for each dataset.
    """

    def __init__(self, datasets, samples_per_epoch, dataset_weights=None):
        if not dataset_weights:  # set default weights
            dataset_weights = [len(d) for d in datasets]

        assert len(datasets) == len(dataset_weights), (
            "Each dataset must have a corresponding weight."
        )

        assert all(w >= 0 for w in dataset_weights), (
            "All dataset weights must be non-negative!"
        )

        total_weight = sum(dataset_weights)
        assert total_weight > 0, "At least one dataset weight has to be larger than 0!"

        self.normalized_weights = np.asarray(
            [w / total_weight for w in dataset_weights]
        )

        self.datasets = datasets
        self.samples_per_epoch = samples_per_epoch

        self._log_summary()

    def _log_summary(self):
        """
        Constructs a formatted summary table string for datasets and weights.
        """
        if len(self.datasets) != len(self.normalized_weights):
            raise ValueError(
                f"Mismatch: {len(self.datasets)} datasets vs {len(self.normalized_weights)} weights"
            )

        # 1. Header
        header = f"{'Idx':<5} | {'Type':<20} | {'Length':>10} | {'Weight':>8}"
        separator = "-" * len(header)
        lines = [header, separator]

        # 2. Rows
        total_len = 0
        for i, (ds, w) in enumerate(zip(self.datasets, self.normalized_weights)):
            d_type = type(
                ds
            ).__name__  # Gets class name (e.g., 'list', 'TensorDataset')
            d_len = len(ds)
            total_len += d_len

            # Formatting: < left align, > right align, .2f for float precision
            line = f"{i:<5} | {d_type:<20} | {d_len:>10,} | {w:>8.2f}"
            lines.append(line)

        # 3. Footer (Optional but helpful)
        lines.append(separator)
        lines.append(
            f"TOTAL DATASETS: {len(self.datasets)} | TOTAL SAMPLES: {total_len:,}"
        )

        log = "\n".join(lines)

        logger.info(log)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # This loop prevents training interruptions when having occasional problems with file loading
        for _ in range(3):
            try:
                ds_idx = np.random.choice(len(self.datasets), p=self.normalized_weights)
                ds = self.datasets[ds_idx]

                sample_idx = np.random.randint(low=0, high=len(ds))

                return ds[sample_idx]
            except Exception as e:
                print(e)
                continue
        else:
            raise ValueError("Too much audio loading repetitions")


class WeightedSequentialConcatDataset(WeightedConcatDataset):
    """
    A PyTorch Dataset that enables weighted sampling from multiple datasets.
    Each dataset is sampled based on specified weights, allowing more frequent
    sampling from certain datasets. Uses worker-aware sampling to avoid duplicates
    across multiple DataLoader workers.

    Args:
        datasets (list of Dataset): List of PyTorch Dataset objects to sample from.
        samples_per_epoch (int): Total number of samples to draw per epoch.
        num_workers (int): The number of workers to be used by the DataLoader
        dataset_weights (list of float, optional): List of weights for each dataset.
    """

    def __init__(self, datasets, samples_per_epoch, num_workers, dataset_weights=None):
        super().__init__(
            datasets=datasets,
            samples_per_epoch=samples_per_epoch,
            dataset_weights=dataset_weights,
        )

        self.num_workers = num_workers

        if self.num_workers == 1:  # if there is one worker it does not have an id
            self.dataset_samplers = [
                cycle(np.random.permutation(len(dataset))) for dataset in self.datasets
            ]
        else:  # each worker should have its own sampler
            self.worker_samplers = {
                worker_id: [
                    cycle(np.random.permutation(len(dataset)))
                    for dataset in self.datasets
                ]
                for worker_id in range(num_workers)
            }

    def __getitem__(self, index):
        worker_info = get_worker_info()

        # This loop prevents training interruptions when having occasional problems with file loading
        for _ in range(3):
            try:
                ds_idx = np.random.choice(len(self.datasets), p=self.normalized_weights)
                ds = self.datasets[ds_idx]

                sample_idx = (
                    next(self.worker_samplers[worker_info.id][ds_idx])
                    if self.num_workers > 1
                    else next(self.dataset_samplers[ds_idx])
                )

                return ds[sample_idx]
            except Exception as e:
                print(e)
                continue
        else:
            raise ValueError("Too much audio loading repetitions")
