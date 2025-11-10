"""Helpers for validating config-driven training builds."""
from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import List, Optional


@dataclass
class DataSourceConfig:
    """Normalized view of a dataset source definition."""

    name: Optional[str] = None
    path: Optional[str] = None
    split: str = "train"
    weight: float = 1.0
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name and not self.path:
            raise ValueError("Each data source must define either 'name' or 'path'.")
        if self.name and self.path:
            raise ValueError("Specify only one of 'name' or 'path' per data source.")
        if self.weight <= 0:
            raise ValueError("Data source weights must be positive.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")


def extract_template_fields(template: str) -> List[str]:
    """Return sorted unique placeholder fields referenced in a format string."""

    fields = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
    return sorted(fields)


__all__ = ["DataSourceConfig", "extract_template_fields"]
