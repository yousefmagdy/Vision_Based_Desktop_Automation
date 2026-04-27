"""Screenshot capture and annotation utilities."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

try:
    import pyautogui
except ImportError:
    pyautogui = None


def capture_screenshot() -> Image.Image:
    """Capture a full screenshot of the desktop.

    Returns:
        PIL Image of the current screen.

    Raises:
        RuntimeError: If screenshot capture fails.
    """
    try:
        if pyautogui is None:
            raise RuntimeError("pyautogui is not installed")
        screenshot = pyautogui.screenshot()
        logger.debug(f"Captured screenshot: {screenshot.size}")
        return screenshot
    except Exception as e:
        logger.error(f"Failed to capture screenshot: {e}")
        raise RuntimeError(f"Screenshot capture failed: {e}") from e


def save_screenshot(
    image: Image.Image,
    path: str | Path,
    annotation: str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    center: tuple[int, int] | None = None,
) -> Path:
    """Save a screenshot with optional annotation overlay.

    Args:
        image: The screenshot image.
        path: File path to save to.
        annotation: Optional text annotation to draw.
        bbox: Optional bounding box (x1, y1, x2, y2) to draw.
        center: Optional center point (x, y) to mark.

    Returns:
        The path where the image was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Draw bounding box with a thick red border
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline="red",
            )

    if center is not None:
        cx, cy = center
        radius = 12
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline="lime",
            width=3,
        )
        # Crosshair
        draw.line([cx - radius - 5, cy, cx + radius + 5, cy], fill="lime", width=2)
        draw.line([cx, cy - radius - 5, cx, cy + radius + 5], fill="lime", width=2)

    if annotation is not None:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Draw text with background
        text_bbox = draw.textbbox((0, 0), annotation, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x, text_y = 10, 10
        draw.rectangle(
            [text_x - 2, text_y - 2, text_x + text_w + 6, text_y + text_h + 6],
            fill="black",
        )
        draw.text((text_x + 2, text_y + 2), annotation, fill="yellow", font=font)

    annotated.save(str(path))
    logger.info(f"Saved screenshot to {path}")
    return path


def capture_and_save(
    save_path: str | Path,
    annotation: str | None = None,
) -> tuple[Image.Image, Path]:
    """Capture a screenshot and save it.

    Returns:
        Tuple of (PIL Image, saved path).
    """
    img = capture_screenshot()
    path = save_screenshot(img, save_path, annotation=annotation)
    return img, path
