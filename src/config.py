"""Load project configuration from config/settings.yaml (pydantic schema).

Qubit counts, depths, dataset sizes, and seeds live in the YAML file, not in
notebooks or library code, so every figure and assertion is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class SimSettings(BaseModel):
    max_qubits_demo: int = 16


class DataSettings(BaseModel):
    n_samples: int = 200
    noise: float = 0.2
    test_size: float = 0.3


class BarrenSettings(BaseModel):
    qubit_counts: list[int] = Field(default_factory=lambda: [2, 4, 6])
    n_layers: int = 10
    n_samples: int = 120


class Settings(BaseModel):
    seed: int = 42
    sim: SimSettings = Field(default_factory=SimSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    barren: BarrenSettings = Field(default_factory=BarrenSettings)


def load_settings(path: Path | str | None = None) -> Settings:
    """Read settings from YAML; fall back to defaults if the file is absent."""
    p = Path(path) if path is not None else _DEFAULT_PATH
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
        return Settings(**data)
    return Settings()
