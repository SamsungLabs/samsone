from dataclasses import dataclass
import json

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from samsone.data.preprocessing.text import TextPreprocessor


@dataclass
class TextDataConfig:
    tokenizer_hf_path: str
    prompt_template_filepath: str
    text_preprocessor: TextPreprocessor | None = None
    response_prefix: str = ""
    prune_tokens_map_path: str | None = None


@dataclass
class QATextData:
    pre_audio_tokens: list[int]
    post_audio_tokens: list[int]
    response_mask: list[int]


class QATextDataset(Dataset):
    def __init__(
        self,
        prompts: list[str],
        responses: list[str],
        tokenizer: AutoTokenizer,
        prompt_template_filepath: str,
        text_preprocessor: TextPreprocessor | None = None,
        response_prefix: str = "",
        prune_tokens_map_path: str | None = None,
    ):
        assert len(prompts) == len(responses), (
            "There should be an equal number of prompts and responses!"
        )
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.text_preprocessor = text_preprocessor
        self.response_prefix = response_prefix

        with open(prompt_template_filepath, "r", encoding="utf-8") as f:
            prompt_template = json.load(f)

        self.system_message = prompt_template["system_prompt"]
        self.delimiter = (
            prompt_template["delimiter"] if self.tokenizer.chat_template else None
        )

        if prune_tokens_map_path is not None:
            with open(prune_tokens_map_path, "r") as tokens_file:
                tokens_map = json.load(tokens_file)
                self.reverse_map = {int(v): int(k) for k, v in tokens_map.items()}
        else:
            self.reverse_map = None

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index: int) -> QATextData:
        prompt, response = self.prompts[index], self.responses[index]

        if self.text_preprocessor:
            prompt = self.text_preprocessor(prompt)
            response = self.text_preprocessor(response)

        pre_audio_text, post_audio_text = self.generate_prompt(prompt)

        post_audio_text += self.response_prefix

        pre_audio_tokens = self.tokenizer(pre_audio_text)["input_ids"]
        post_audio_tokens_list = self.tokenizer(
            [post_audio_text, response + self.tokenizer.eos_token]
        )["input_ids"]
        response_mask = [0] * len(post_audio_tokens_list[0]) + [1] * len(
            post_audio_tokens_list[1]
        )
        post_audio_tokens = [
            tok for sublist in post_audio_tokens_list for tok in sublist
        ]

        if self.reverse_map is not None:
            post_audio_tokens = [self.reverse_map[tok] for tok in post_audio_tokens]
            pre_audio_tokens = [self.reverse_map[tok] for tok in pre_audio_tokens]

        return QATextData(
            pre_audio_tokens=pre_audio_tokens,
            post_audio_tokens=post_audio_tokens,
            response_mask=response_mask,
        )

    def generate_prompt(self, prompt):
        if not self.tokenizer.chat_template:
            return self.system_message, prompt

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        split_index = text.find(self.delimiter) + len(self.delimiter)

        pre_audio_text = text[:split_index]
        post_audio_text = text[split_index:]

        return pre_audio_text, post_audio_text
