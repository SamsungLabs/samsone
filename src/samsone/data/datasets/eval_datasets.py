from torch.utils.data import Dataset

from samsone.evaluation import ALMEvaluator


class ALMEvalDataset(Dataset):
    def __init__(self, dataset: Dataset, evaluator: ALMEvaluator):
        self.dataset = dataset
        self.evaluator = evaluator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
