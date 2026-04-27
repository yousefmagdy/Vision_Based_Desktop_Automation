"""CLI entry point using Typer."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="vision-auto",
    help="Vision-based desktop automation with dynamic icon grounding.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        "automation.log",
        level="DEBUG",
        rotation="5 MB",
        retention="3 days",
    )


@app.command()
def run(
    output_dir: str = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory to save post files. Default: ~/Desktop/tjm-project",
    ),
    post_count: int = typer.Option(
        10,
        "--posts", "-n",
        help="Number of posts to process (1-100).",
        min=1,
        max=100,
    ),
    model: str = typer.Option(
        "microsoft/Florence-2-large",
        "--model", "-m",
        help="Florence-2 model name or path.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Compute device: 'cuda' or 'cpu'.",
    ),
    icon_label: str = typer.Option(
        "Notepad",
        "--icon", "-i",
        help="Label of the desktop icon to find.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run the full Notepad automation workflow.

    Captures screenshots, locates the Notepad icon using Florence-2 visual grounding,
    launches Notepad, types and saves blog posts from JSONPlaceholder API.
    """
    _setup_logging(verbose)

    from .config import Config
    from .automation import NotepadAutomation

    config = Config(
        model_name=model,
        device=device,
        post_count=post_count,
        notepad_icon_label=icon_label,
    )
    if output_dir:
        config.output_dir = output_dir

    console.print(Panel.fit(
        "[bold green]Vision-Based Desktop Automation[/]\n"
        f"Model: {config.model_name}\n"
        f"Device: {config.device}\n"
        f"Posts: {config.post_count}\n"
        f"Target icon: {config.notepad_icon_label}\n"
        f"Output: {config.output_dir}",
        title="Configuration",
    ))

    automation = NotepadAutomation(config)
    automation.run()


@app.command()
def demo_screenshots(
    model: str = typer.Option(
        "microsoft/Florence-2-large",
        "--model", "-m",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
    ),
) -> None:
    """Generate the 3 required annotated demo screenshots.

    Interactive mode: prompts you to move the Notepad icon to
    top-left, bottom-right, and center positions.
    """
    _setup_logging(verbose)

    from .config import Config
    from .automation import create_demo_screenshots

    config = Config(model_name=model, device=device)

    console.print(Panel.fit(
        "[bold cyan]Demo Screenshot Generator[/]\n"
        "This will capture 3 annotated screenshots showing\n"
        "icon detection at different desktop positions.\n\n"
        "You will be prompted to move the Notepad icon.",
        title="Demo Mode",
    ))

    create_demo_screenshots(config)
    console.print("[bold green]✓ Demo screenshots saved to screenshots/ directory[/]")


@app.command()
def ground(
    target: str = typer.Argument(
        help="Text description of the element to find (e.g., 'Notepad', 'Chrome').",
    ),
    model: str = typer.Option(
        "microsoft/Florence-2-large",
        "--model", "-m",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
    ),
    save_screenshot: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save an annotated screenshot.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
    ),
) -> None:
    """Ground a single element on the desktop (useful for testing).

    Takes a screenshot, runs the grounding pipeline for the given target,
    and reports the results.
    """
    _setup_logging(verbose)

    from .config import Config
    from .grounding import VisualGrounder
    from .screenshot import capture_screenshot as capture, save_screenshot as save_ss

    config = Config(model_name=model, device=device)
    grounder = VisualGrounder(config)

    console.print(f"[cyan]Grounding target: '{target}'[/]")
    console.print("[dim]Capturing screenshot...[/]")

    screenshot = capture()
    detection = grounder.ground(screenshot, target)

    if detection:
        console.print(Panel.fit(
            f"[bold green]Found: {detection.label}[/]\n"
            f"Center: ({detection.center[0]}, {detection.center[1]})\n"
            f"BBox: {detection.bbox}\n"
            f"Confidence: {detection.confidence:.2f}",
            title="Detection Result",
        ))

        if save_screenshot:
            path = save_ss(
                screenshot,
                Path(config.screenshot_dir) / f"ground_{target.replace(' ', '_')}.png",
                annotation=f"Found: {detection.label} ({detection.confidence:.2f})",
                bbox=detection.bbox,
                center=detection.center,
            )
            console.print(f"[dim]Screenshot saved: {path}[/]")
    else:
        console.print(f"[bold red]✗ Could not find '{target}' on screen[/]")

        if save_screenshot:
            path = save_ss(
                screenshot,
                Path(config.screenshot_dir) / f"ground_{target.replace(' ', '_')}_failed.png",
                annotation=f"NOT FOUND: {target}",
            )
            console.print(f"[dim]Screenshot saved: {path}[/]")


@app.command()
def list_icons(
    model: str = typer.Option(
        "microsoft/Florence-2-large",
        "--model", "-m",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
    ),
) -> None:
    """Detect and list ALL desktop icons (Bonus: multi-icon detection).

    Takes a screenshot and uses OCR to find all icon labels visible
    on the desktop. Useful for debugging and verifying icon detection.
    """
    _setup_logging(verbose)

    from .config import Config
    from .grounding import VisualGrounder
    from .screenshot import capture_screenshot as capture, save_screenshot as save_ss

    config = Config(model_name=model, device=device)
    grounder = VisualGrounder(config)

    console.print("[cyan]Scanning desktop for all icons...[/]")
    screenshot = capture()
    icons = grounder.detect_all_icons(screenshot)

    if icons:
        console.print(f"\n[bold green]Found {len(icons)} icon labels:[/]")
        for i, icon in enumerate(icons, 1):
            console.print(
                f"  {i:2d}. [cyan]{icon.label}[/] "
                f"at ({icon.center[0]}, {icon.center[1]}) "
                f"bbox={tuple(int(v) for v in icon.bbox)}"
            )

        # Save annotated screenshot with all detections
        for icon in icons:
            save_ss(
                screenshot,
                Path(config.screenshot_dir) / "all_icons.png",
                annotation=f"{len(icons)} icons detected",
                bbox=icon.bbox,
                center=icon.center,
            )
        console.print(f"\n[dim]Screenshot saved to screenshots/all_icons.png[/]")
    else:
        console.print("[bold red]No icon labels detected[/]")


if __name__ == "__main__":
    app()
