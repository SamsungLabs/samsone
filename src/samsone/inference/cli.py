"""Console entry points for released SAMSONE inference models."""

import argparse
from collections.abc import Callable

import torch

from .models import Samsone99M, Samsone134M, Samsone356M, SamsoneModel


def _run_cli(model_type: Callable[..., SamsoneModel]) -> None:
    parser = argparse.ArgumentParser(description="Run SAMSONE audio inference.")
    parser.add_argument(
        "--audio", nargs="+", required=True, help="One or more audio files."
    )
    parser.add_argument(
        "--prompt", required=True, help="Prompt to ask about the audio."
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. cuda or cpu."
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Model floating-point dtype (default: bfloat16).",
    )
    parser.add_argument("--checkpoint-path", default=None)
    arguments = parser.parse_args()
    model = model_type(
        device=arguments.device,
        dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[arguments.dtype],
        checkpoint_path=arguments.checkpoint_path,
        max_new_tokens=arguments.max_new_tokens,
    )
    print(model(audio=arguments.audio, prompt=arguments.prompt))


def main_99m() -> None:
    _run_cli(Samsone99M)


def main_134m() -> None:
    _run_cli(Samsone134M)


def main_356m() -> None:
    _run_cli(Samsone356M)
