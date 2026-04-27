"""Configuration constants and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""

    # Display
    screen_width: int = 1920
    screen_height: int = 1080

    # Grounding model
    model_name: str = "microsoft/Florence-2-large"
    device: str = "cuda"  # Will fallback to cpu if cuda unavailable
    confidence_threshold: float = 0.15

    # Cascaded search
    cascade_levels: int = 3
    crop_overlap: float = 0.15
    grid_size: tuple[int, int] = (2, 2)  # For cascaded region splitting

    # Retry logic
    max_retries: int = 3
    retry_delay: float = 1.0

    # Automation timing (seconds)
    launch_wait: float = 2.0
    type_delay: float = 0.02
    action_delay: float = 0.5
    save_dialog_wait: float = 1.0

    # Paths
    output_dir: str = field(default_factory=lambda: os.path.join(
        os.path.expanduser("~"), "Desktop", "tjm-project"
    ))
    screenshot_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "screenshots",
    ))

    # API
    api_base_url: str = "https://jsonplaceholder.typicode.com"
    post_count: int = 10

    # Notepad
    notepad_icon_label: str = "Notepad"
    notepad_window_title: str = "Notepad"

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        try:
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"
