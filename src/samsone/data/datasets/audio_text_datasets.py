from dataclasses import dataclass

from torch.utils.data import Dataset

from samsone.data.datasets.audio_datasets import AudioFeatureData
from samsone.data.datasets.text_datasets import QATextData


@dataclass
class AudioQATextData:
    audio_feature_data: AudioFeatureData
    qa_text_data: QATextData
    unique_id: str


class AudioQATextDataset(Dataset):
    def __init__(
        self,
        audio_dataset: Dataset,
        text_dataset: Dataset,
        ids: list[str],
    ):
        assert len(audio_dataset) == len(text_dataset), (
            "Audio and text datasets should be of equal length!"
        )
        assert len(audio_dataset) == len(ids), (
            "Dataset length and number of ids should be equal!"
        )
        self.audio_dataset = audio_dataset
        self.text_dataset = text_dataset
        self.ids = ids

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, index) -> AudioQATextData:
        audio_feature_data = self.audio_dataset[index]
        qa_text_data = self.text_dataset[index]
        unique_id = self.ids[index]

        return AudioQATextData(
            audio_feature_data=audio_feature_data,
            qa_text_data=qa_text_data,
            unique_id=unique_id,
        )
