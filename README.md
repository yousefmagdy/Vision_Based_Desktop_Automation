# Vision-Based Desktop Automation with Dynamic Icon Grounding

A Python application that uses computer vision (Florence-2 VLM) to dynamically locate and interact with desktop icons on Windows, enabling robust automation even when icon positions change.

## Architecture

```
vision-desktop-automation/
├── pyproject.toml              # Project config (uv/pip compatible)
├── README.md
├── screenshots/                # Annotated detection screenshots
├── tests/
│   ├── test_grounding.py       # Grounding pipeline tests
│   ├── test_api_client.py      # API client tests
│   └── test_screenshot.py      # Screenshot utility tests
└── src/vision_desktop_automation/
    ├── __init__.py
    ├── __main__.py             # python -m entry point
    ├── cli.py                  # Typer CLI (vision-auto)
    ├── config.py               # Centralized configuration
    ├── grounding.py            # Florence-2 visual grounding engine
    ├── screenshot.py           # Screenshot capture & annotation
    ├── desktop.py              # Mouse/keyboard/window management
    ├── automation.py           # Workflow orchestrator
    └── api_client.py           # JSONPlaceholder API client
```

## Grounding Approach

This project implements a **flexible, open-vocabulary visual grounding system** inspired by [ScreenSeekeR](https://arxiv.org/abs/2504.07981) (ScreenSpot-Pro). Instead of template matching against pre-captured icon images, we use **Microsoft Florence-2**, a vision-language model that can locate arbitrary UI elements given a natural-language description.

### Why Florence-2 over Template Matching?

| Aspect | Template Matching | Florence-2 (Our Approach) |
|--------|------------------|--------------------------|
| Requires reference image | Yes | No |
| Handles theme changes | Poorly | Well |
| Works for unknown icons | No | Yes |
| Handles pop-ups/dialogs | No (needs templates) | Yes (describe what to dismiss) |
| Resolution independent | No | Yes |
| Handles partial occlusion | Poorly | Better |

### OCR-First Pipeline

The grounding engine uses an **OCR-first approach** — desktop icons on Windows always have a text label beneath the icon image, making OCR the most reliable detection signal. The pipeline cascades through three phases:

1. **Phase 1 — Full-Image OCR Search**: Runs Florence-2's `<OCR_WITH_REGION>` task on the full screenshot. Matches the target text (case-insensitive) against all detected labels, then expands the bounding box upward to include the icon image above the text. Confidence: 0.95.

2. **Phase 2 — Cascaded OCR Search**: If Phase 1 misses, the screenshot is split into a 2×2 overlapping grid. OCR runs independently on each sub-region at higher effective resolution, then coordinates are mapped back to full-image space. This catches small text on high-res displays.

3. **Phase 3 — Phrase Grounding + OCR Verification**: As a final fallback, Florence-2's `CAPTION_TO_PHRASE_GROUNDING` and `OPEN_VOCABULARY_DETECTION` tasks are used with multiple query variations. Every detection is **verified by running OCR** on the region around it — if the target text isn't found nearby, the detection is discarded. This prevents hallucinations.

4. **NMS + Region Biasing**: Non-maximum suppression removes duplicate detections. Optional region hints (e.g., "top-left") bias the selection without hard-filtering.

## Prerequisites

- **OS**: Windows 10/11 at 1920x1080 resolution
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA recommended (CPU fallback available, slower)
- **Desktop**: Notepad shortcut icon placed on the desktop

## Setup

### Using uv (recommended)

```bash
# Install uv if you don't have it
pip install uv

# Clone and install
git clone <repo-url>
cd vision-desktop-automation
uv sync

# Or with dev dependencies
uv sync --extra dev
```

### Using pip

```bash
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### PyTorch with CUDA

If you have an NVIDIA GPU, install PyTorch with CUDA support first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Full Automation Run

```bash
# Default: process 10 posts, save to ~/Desktop/tjm-project/
vision-auto run

# Custom options
vision-auto run --posts 5 --output-dir C:\Users\me\Desktop\output --device cpu -v
```

### Generate Demo Screenshots (Deliverable)

```bash
# Interactive: prompts you to move the Notepad icon to 3 positions
# Produces 3 annotated screenshots in screenshots/ directory
vision-auto demo-screenshots
```

### Test Grounding on Any Element

```bash
# Ground a specific element (useful for testing/debugging)
vision-auto ground "Notepad"
vision-auto ground "Chrome" --device cpu
vision-auto ground "Recycle Bin" -v
```

### List All Desktop Icons (Bonus)

```bash
# Detect and list every icon on the desktop in one pass
vision-auto list-icons
```

### Run as Module

```bash
python -m vision_desktop_automation run
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/test_grounding.py -v
```

## Error Handling

The system handles these failure scenarios:

- **Icon not found**: Retries up to 3 times with 1s delay between attempts
- **Notepad fails to launch**: Validates window title within timeout; retries full minimize→ground→launch sequence up to 3 times
- **User switches tabs during automation**: Detects that Notepad didn't open, re-minimizes all windows, and retries from desktop
- **API unavailable**: Falls back to generated placeholder posts
- **Existing files**: Overwrites with confirmation handling in Save dialog
- **Multiple matching icons**: `select_best_match()` disambiguates using exact → prefix → substring matching
- **Partially obscured icons**: Cascaded search on sub-regions can still detect visible portions
- **Window management**: 3-strategy minimize (Shell.Application COM → Win+M → Win+D) with foreground verification
- **Desktop background interference**: OCR-based detection reads text labels, not pixel patterns — inherently theme-agnostic

## Configuration

Key settings in `Config` (configurable via CLI flags):

| Setting | Default | Description |
|---------|---------|-------------|
| `model_name` | `microsoft/Florence-2-large` | Florence-2 model variant |
| `device` | `cuda` (auto-fallback to `cpu`) | Compute device |
| `confidence_threshold` | `0.15` | Minimum detection confidence |
| `cascade_levels` | `3` | Max depth for cascaded search |
| `max_retries` | `3` | Retry attempts for icon detection |
| `retry_delay` | `1.0` | Seconds between retries |
| `post_count` | `10` | Number of posts to process |

## Discussion Points

### Why Florence-2?

Florence-2 provides open-vocabulary grounding — you describe what you want to find in natural language rather than providing a reference image. This makes the system inherently flexible: it can find any icon, button, or UI element without prior knowledge of its appearance. This directly addresses the requirement that the system should work "for any icon or button, even if we don't have the exact image or text beforehand."

### Failure Cases

Detection may fail when:
- Icons have no text label (renamed or taskbar-pinned icons without visible text)
- Icons are extremely small (e.g., small icon view + 4K DPI) — cascaded search mitigates this
- Two icons have overlapping text labels (placed very close together)
- User interacts with the desktop during the grounding-to-click pipeline

Mitigations: OCR-first pipeline, cascaded sub-region search, OCR verification of phrase grounding, retry logic.

### Performance

- Model loading: ~10-30s (one-time, cached for session)
- Per-detection Phase 1 (GPU): ~1-2s
- Per-detection Phase 1 (CPU): ~50-60s
- Per-detection Phase 2 cascaded (CPU): ~3-5 minutes
- Full 10-post workflow (CPU): ~15-20 minutes
- Full 10-post workflow (GPU): ~3-5 minutes

### Scaling

The architecture naturally extends to:
- **Any icon**: Change the `--icon` flag to any description
- **All icons at once**: `vision-auto list-icons` detects every icon in one pass
- **Pop-up handling**: Use `grounder.ground(screenshot, "OK button")` or `"Close dialog"`
- **Different resolutions**: Florence-2 processes at model resolution, independent of screen size; increase `grid_size` for 4K+
- **Multi-monitor**: Capture specific monitor screenshots and offset coordinates

### Bonus Challenges

1. **Multi-icon detection**: `detect_all_icons()` + `select_best_match()` — detects all icons and disambiguates similar names (Notepad vs Notepad++)
2. **Size-independent**: Cascaded OCR search + dynamic bbox expansion handles small/medium/large icon settings
3. **Theme-agnostic**: OCR-based detection reads text labels, works across light/dark themes and any wallpaper

## License

MIT
