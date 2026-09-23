from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_utils import no_init_weights

from samsone.models.base import Model
from samsone.models.text.utils import prune_input_vocab


@dataclass
class GenerationConfig:
    do_sample: bool = False
    temperature: float = 0
    top_p: float = 1.0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1
    early_stopping: bool = False


class LM(Model, ABC):
    max_tokens: bool = None

    @abstractmethod
    def get_embeddings_from_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def process_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        pass

    @abstractmethod
    def forward(
        self, x: Any, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> Any:
        pass

    def tokenize(self, text: list[str], device) -> torch.Tensor:
        tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=self.max_tokens
        )

        for key, value in tokens.items():
            tokens[key] = value.to(device)

        return tokens

    @abstractmethod
    def generate(
        self, input_embeddings, attention_mask, max_new_tokens, **kwargs
    ) -> list[str]:
        pass


class HuggingFaceLM(LM):
    def __init__(
        self,
        model_name_or_path: str,
        generation_config: GenerationConfig = GenerationConfig(),
        num_hidden_layers: int | None = None,
        prune_tokens_map_path: str | None = None,
        load_pretrained_weights: bool = True,
    ):
        super().__init__()
        if load_pretrained_weights:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            )
        else:  # weights will come from a Samsone checkpoint
            with no_init_weights():
                self.model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(model_name_or_path)
                )
        if num_hidden_layers is not None:
            if self.model.config.num_hidden_layers < num_hidden_layers:
                raise ValueError(
                    f"{model_name_or_path} has {self.model.config.num_hidden_layers} hidden layers and {num_hidden_layers} hidden layers has been selected!"
                )
            self.model.config.num_hidden_layers = num_hidden_layers
            self.model.model.layers = self.model.model.layers[
                :num_hidden_layers
            ]  # this was tested only on SmolLM2
            del self.model.model.layers[num_hidden_layers:]
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        self.generation_config = generation_config

        if prune_tokens_map_path is not None:
            with open(prune_tokens_map_path, "r") as tokens_file:
                tokens_map = json.load(tokens_file)

            self.tokens_map = {int(k): int(v) for k, v in tokens_map.items()}
            self.model.config.vocab_size = len(tokens_map)
            self.model.model = prune_input_vocab(
                self.model.model, tokens_map, "embed_tokens"
            )
            self.model.tie_weights()

        self.embedding_layer = self.model.get_input_embeddings()

        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def get_embeddings_from_tokens(self, input_tokens: torch.Tensor) -> torch.Tensor:
        # The result is a tensor of shape [batch_size, sequence_length, embedding_dim]
        input_tokens = input_tokens.long()
        initial_embeddings = self.embedding_layer(input_tokens)

        return initial_embeddings

    def process_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.model(
            inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        initial_embeddings = self.get_embeddings_from_tokens(input_tokens)
        output = self.process_embeddings(initial_embeddings, attention_mask, labels)

        return output

    def generate(
        self,
        input_embeddings,
        attention_mask,
        max_new_tokens,
        **kwargs,
    ) -> list[str]:
        generation_kwargs = asdict(self.generation_config)
        if not generation_kwargs["do_sample"]:
            generation_kwargs.pop("temperature")
            generation_kwargs.pop("top_p")

        generated_ids = self.model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
            **generation_kwargs,
        )
        if hasattr(self, "tokens_map"):
            generated_ids = [
                [self.tokens_map[i.item()] for i in ids] for ids in generated_ids
            ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def decode(self, tokens: list[torch.Tensor]):
        if hasattr(self, "tokens_map"):
            tokens = [self.tokens_map[token.item()] for token in tokens]

        return self.tokenizer.decode(tokens)
