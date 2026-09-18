from abc import ABC, abstractmethod
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable

import torch.nn as nn


@dataclass
class FreezeConfig:
    freeze_modules: list[str] | None = None


class Model(nn.Module, ABC):
    """
    Base model class that extends PyTorch's nn.Module with freeze/unfreeze functionality.
    All model classes should inherit from this class to get consistent parameter freezing capabilities.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def freeze(self, freeze_modules: list[str]):
        """
        Freeze params of matching submodules.
        """
        resolved_modules = self._resolve_modules(freeze_modules)

        for m in resolved_modules:
            for p in m.parameters(recurse=True):
                p.requires_grad = False
                m.eval()

    def unfreeze(self, freeze_modules: list[str]):
        """
        Unfreeze params of matching submodules.
        """
        resolved_modules = self._resolve_modules(freeze_modules)

        for m in resolved_modules:
            for p in m.parameters(recurse=True):
                p.requires_grad = True
                m.train()

    def get_trainable_parameters(self) -> tuple:
        return (p for p in self.parameters() if p.requires_grad)

    def get_num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_parameters(self) -> tuple:
        return (p for p in self.parameters() if not p.requires_grad)

    def get_num_frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def get_num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def freeze_summary(self, depth: int = 1):
        """
        Prints a hierarchical summary of the model's freeze status.

        Args:
            depth (int): The maximum depth of submodules to display.
                         -1 will display all levels.
                         1 (default) will show only the top-level modules.
        """
        if depth == 0:
            return

        print("\n" + "=" * 85)
        print(" " * 32 + "Freeze Summary")
        print("=" * 85)
        print(f"{'Module Name':<40} {'Status':<15} {'Trainable':>15} {'Total':>15}")
        print("-" * 85)

        for name, module in self.named_children():
            self._print_summary_recursive(
                module, name, current_depth=1, max_depth=depth
            )

        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        print("-" * 85)
        print(
            f"{'GRAND TOTAL':<40} {'':<15} {total_trainable_params:>15,} {total_params:>15,}"
        )
        print("=" * 85 + "\n")

    def _print_summary_recursive(
        self,
        module: nn.Module,
        name: str,
        current_depth: int,
        max_depth: int,
        indent: str = "",
    ):
        """Recursively prints the summary for a module and its children."""

        module_params = list(module.parameters())
        num_params = sum(p.numel() for p in module_params)
        num_trainable = sum(p.numel() for p in module_params if p.requires_grad)

        status = "No Parameters"
        if num_params > 0:
            if num_trainable == 0:
                status = "Frozen ❄️"
            elif num_trainable == num_params:
                status = "Trainable 🔥"
            else:
                status = "Partial 🔥/❄️"

        print(
            f"{indent + name:<40} {status:<15} {num_trainable:>15,} {num_params:>15,}"
        )

        if max_depth != -1 and current_depth >= max_depth:
            return

        for child_name, child_module in module.named_children():
            self._print_summary_recursive(
                child_module,
                child_name,
                current_depth + 1,
                max_depth,
                indent="  " + indent,
            )

    def _resolve_modules(self, names: Iterable[str] | None) -> list[nn.Module]:
        """
        Resolve module names and glob patterns against named_modules().

        Args:
            names: An iterable of strings (dotted names or glob patterns)
                   to match against module-qualified names.
                   If None or empty, returns an empty list.

        Returns:
            A list of unique nn.Module instances that matched the patterns.

        Raises:
            ValueError: If any pattern in 'names' does not match at least one module.
        """
        if not names:  # freeze nothing
            return []

        patterns = list(names)
        result = set()
        # Build a dict of "qualified name" -> module
        qmap = dict(self.named_modules())  # includes "" for self

        # Normalize: Map "" (self) to its class name for matching
        if "" in qmap:
            qmap[self.__class__.__name__] = qmap.pop("")

        # Get all available module names for matching and error messages
        available_names = set(qmap.keys())

        for pat in patterns:
            pattern_matched = False

            for qname in available_names:
                if self._matches(qname, pat):
                    result.add(qmap[qname])
                    pattern_matched = True

            # failsafe for typos
            if not pattern_matched:
                raise ValueError(
                    f"The pattern '{pat}' did not match any submodules. "
                    f"Available examples: {list(self._sample_names(available_names))[:10]}"
                )

        return list(result)

    def _matches(self, name: str, pattern: str) -> bool:
        """Helper for checking glob patterns and exact matches."""
        return name == pattern or fnmatch(name, pattern)

    def _sample_names(self, names: Iterable[str]) -> Iterable[str]:
        """Helper for generating example names for the error message."""
        for n in names:
            if n:
                yield n
