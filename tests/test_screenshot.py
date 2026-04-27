"""Tests for screenshot utilities."""

from __future__ import annotations

import os
import tempfile

import pytest
from PIL import Image

from vision_desktop_automation.screenshot import save_screenshot


class TestSaveScreenshot:
    """Tests for save_screenshot function."""

    def test_save_basic(self, tmp_path) -> None:
        img = Image.new("RGB", (1920, 1080), color="blue")
        path = save_screenshot(img, tmp_path / "test.png")
        assert path.exists()
        saved = Image.open(path)
        assert saved.size == (1920, 1080)

    def test_save_with_annotation(self, tmp_path) -> None:
        img = Image.new("RGB", (800, 600), color="black")
        path = save_screenshot(
            img, tmp_path / "annotated.png", annotation="Test annotation"
        )
        assert path.exists()

    def test_save_with_bbox(self, tmp_path) -> None:
        img = Image.new("RGB", (800, 600), color="white")
        path = save_screenshot(
            img, tmp_path / "bbox.png", bbox=(100, 100, 200, 200)
        )
        assert path.exists()

    def test_save_with_center(self, tmp_path) -> None:
        img = Image.new("RGB", (800, 600), color="gray")
        path = save_screenshot(
            img, tmp_path / "center.png", center=(400, 300)
        )
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path) -> None:
        img = Image.new("RGB", (100, 100), color="red")
        path = save_screenshot(img, tmp_path / "sub" / "dir" / "test.png")
        assert path.exists()
        assert path.parent.name == "dir"
