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

This project uses **Microsoft Florence-2**, an open-vocabulary vision-language model, to locate UI elements via natural language descriptions instead of static template matching.

### OCR-First Pipeline
Desktop icons on Windows typically have a text label, making OCR the most reliable detection signal. The pipeline follows a cascaded search:
1. **Full-Image OCR**: Locates the text label and expands the target area to include the icon.
2. **Cascaded Grid OCR**: Splits the screen into overlapping regions to detect small text on high-resolution displays.
3. **Phrase Grounding Fallback**: Uses visual grounding with OCR verification to prevent false positives.

## Prerequisites

- **OS**: Windows 10/11
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA recommended (CPU fallback available)
- **Desktop**: Notepad shortcut icon present on the desktop

## Setup

### Using uv (recommended)
```bash
uv sync
```

### Using pip
```bash
pip install -e .
```

## Usage

### Full Automation Run
```bash
vision-auto run
```

### Generate Demo Screenshots
```bash
vision-auto demo-screenshots
```

### Test Grounding
```bash
vision-auto ground "Notepad"
```

## Error Handling

The system is robust against:
- **Icon not found**: Automatic retries with delays.
- **Launch failure**: Window title validation and full sequence retries.
- **API unavailable**: Falls back to generated placeholder posts.
- **Existing files**: Handles "Replace" confirmation dialogs.
- **Occlusion/Theme**: OCR-based detection is inherently theme-agnostic and resolution-independent.

## License
MIT
