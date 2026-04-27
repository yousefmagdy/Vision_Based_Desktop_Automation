"""Main automation workflow orchestrator.

Implements the full automation pipeline:
1. Capture screenshot → ground Notepad icon → launch
2. Fetch posts from API
3. For each post: type content → save as file → close Notepad → repeat
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from .api_client import ApiClient, Post
from .config import Config
from .desktop import (
    close_window,
    dismiss_popups,
    double_click,
    focus_window,
    hotkey,
    is_window_open,
    kill_process,
    list_visible_windows,
    minimize_all_windows,
    press_key,
    single_click,
    type_text,
    wait_for_window,
)
from .grounding import Detection, VisualGrounder
from .screenshot import capture_screenshot, save_screenshot


class AutomationError(Exception):
    """Raised when a critical automation step fails."""


class NotepadAutomation:
    """Orchestrates the full Notepad automation workflow."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.grounder = VisualGrounder(self.config)
        self.api_client = ApiClient(self.config.api_base_url)
        self._screenshot_counter = 0

    def run(self) -> None:
        """Execute the full automation workflow.

        Steps:
            1. Pre-flight: ensure output directory exists, load model
            2. Fetch posts from API
            3. For each post:
               a. Ensure desktop is visible
               b. Capture screenshot
               c. Ground Notepad icon
               d. Double-click to launch
               e. Type and save post content
               f. Close Notepad
        """
        logger.info("=" * 60)
        logger.info("Starting Vision-Based Desktop Automation")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 60)

        # Pre-flight
        self._preflight()

        # Fetch posts
        posts = self._fetch_posts()
        if not posts:
            logger.error("No posts retrieved. Aborting.")
            return

        logger.info(f"Processing {len(posts)} posts...")

        # Process each post
        for i, post in enumerate(posts):
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Post {i + 1}/{len(posts)}: ID={post.id}")
            logger.info(f"Title: {post.title[:60]}...")

            try:
                self._process_single_post(post)
                logger.info(f"✓ Post {post.id} saved successfully")
            except AutomationError as e:
                logger.error(f"✗ Failed to process post {post.id}: {e}")
                # Try to recover by closing any open Notepad windows
                self._emergency_cleanup()
            except Exception as e:
                logger.error(f"✗ Unexpected error processing post {post.id}: {e}")
                self._emergency_cleanup()

        logger.info("\n" + "=" * 60)
        logger.info("Automation complete!")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 60)

    def _preflight(self) -> None:
        """Pre-flight checks and setup."""
        # Ensure output dir exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Pre-load the grounding model (so first iteration isn't slow)
        logger.info("Loading grounding model (this may take a moment)...")
        self.grounder.load_model()
        logger.info("Model ready.")

    def _fetch_posts(self) -> list[Post]:
        """Fetch posts with graceful error handling."""
        try:
            return self.api_client.fetch_posts(self.config.post_count)
        except ConnectionError as e:
            logger.error(f"API unavailable: {e}")
            logger.info("Generating placeholder posts as fallback...")
            return self._generate_fallback_posts()

    def _generate_fallback_posts(self) -> list[Post]:
        """Generate fallback posts if the API is unavailable."""
        return [
            Post(
                id=i,
                user_id=1,
                title=f"Fallback Post {i}",
                body=f"This is a fallback post generated because the API was unavailable.\n"
                     f"Post number: {i}",
            )
            for i in range(1, self.config.post_count + 1)
        ]

    def _process_single_post(self, post: Post) -> None:
        """Process a single post through the full automation cycle.

        Robust against the user switching windows/tabs during automation:
        if Notepad fails to launch, re-minimizes everything and retries
        the full show-desktop → ground → launch sequence.

        Args:
            post: The Post object to process.

        Raises:
            AutomationError: If any step fails after all retries.
        """
        max_launch_attempts = 3

        for attempt in range(1, max_launch_attempts + 1):
            # Step 1: Show the desktop (minimize all windows)
            self._show_desktop()

            # Step 2: Capture screenshot and ground Notepad icon
            detection = self._find_and_launch_notepad()

            # Step 3: Wait for Notepad to open
            if wait_for_window("Notepad", timeout=10.0):
                # Critical: Dismiss any "Cannot find file" restoration popups
                dismiss_popups()
                time.sleep(0.5)
                break  # Success — Notepad is open

            # Notepad didn't open. The user may have switched tabs/windows,
            # causing the double-click to land on the wrong window.
            windows = list_visible_windows()
            logger.warning(
                f"Attempt {attempt}/{max_launch_attempts}: Notepad didn't open. "
                f"Foreground windows: {[w for w in windows[:5] if 'Progman' not in w]}"
            )

            if attempt < max_launch_attempts:
                logger.info("Re-minimizing and retrying full sequence...")
                time.sleep(1.0)
            else:
                raise AutomationError(
                    f"Notepad did not open after {max_launch_attempts} attempts. "
                    "Ensure the desktop is clear and the Notepad icon is accessible."
                )

        time.sleep(self.config.action_delay)

        # Step 4: Focus Notepad and type content
        focus_window("Notepad")
        time.sleep(self.config.action_delay)

        content = post.format_content()
        # Clear any existing content (Select All). Overwriting happens during the paste in type_text.
        hotkey("ctrl", "a")
        time.sleep(0.1)
        type_text(content, interval=self.config.type_delay)

        # Step 5: Save the file
        filename = f"post_{post.id}.txt"
        self._save_file(filename)

        # Step 6: Close Notepad
        self._close_notepad()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1.0),
        retry=retry_if_exception_type(AutomationError),
        reraise=True,
    )
    def _find_and_launch_notepad(self) -> Detection:
        """Find the Notepad icon and double-click it.

        Implements retry logic: up to 3 attempts with 1s delay.

        Returns:
            The Detection object for the found icon.

        Raises:
            AutomationError: If icon cannot be found after retries.
        """
        # Capture fresh screenshot
        screenshot = capture_screenshot()
        self._screenshot_counter += 1

        # Save annotated screenshot for deliverables
        screenshot_path = os.path.join(
            self.config.screenshot_dir,
            f"screenshot_{self._screenshot_counter:03d}.png",
        )

        # Ground the Notepad icon
        detection = self.grounder.ground(screenshot, self.config.notepad_icon_label)

        if detection is None:
            # Save screenshot showing failure
            save_screenshot(
                screenshot,
                screenshot_path,
                annotation=f"FAILED: Could not find '{self.config.notepad_icon_label}'",
            )
            raise AutomationError(
                f"Could not locate '{self.config.notepad_icon_label}' icon on desktop"
            )

        # Save annotated screenshot with detection
        save_screenshot(
            screenshot,
            screenshot_path,
            annotation=(
                f"Found: {detection.label} at ({detection.center[0]}, {detection.center[1]}) "
                f"conf={detection.confidence:.2f}"
            ),
            bbox=detection.bbox,
            center=detection.center,
        )

        # Double-click the icon
        logger.info(f"Launching Notepad at ({detection.center[0]}, {detection.center[1]})")
        double_click(*detection.center)

        return detection

    def _show_desktop(self) -> None:
        """Ensure the desktop is visible.

        Uses minimize_all_windows() which tries Shell.Application COM,
        Win+M, and Win+D in sequence. Then verifies by checking if any
        non-desktop windows remain in the foreground.
        """
        # Forcefully kill Notepad to prevent session restoration / tab issues
        if is_window_open("Notepad"):
            logger.debug("Killing lingering Notepad process...")
            kill_process("Notepad.exe")
            time.sleep(1.0)

        minimize_all_windows()
        time.sleep(self.config.action_delay)

        # Log remaining visible windows for debugging
        windows = list_visible_windows()
        non_system = [
            w for w in windows
            if not any(cls in w for cls in [
                "Progman", "WorkerW", "Shell_TrayWnd",
                "Windows.UI.Core.CoreWindow",
                "Windows Input Experience",
            ])
        ]
        if non_system:
            logger.debug(
                f"After minimize, {len(non_system)} non-system windows remain: "
                f"{non_system[:5]}"
            )

    def _save_file(self, filename: str) -> None:
        """Save the current Notepad content using Save As dialog.

        Handles both classic Notepad and Windows 11 modern Notepad.
        Win11 Notepad uses a modern file picker whose window title may be
        "Save As" or contain "Save". Classic Notepad uses "Save As".

        Args:
            filename: The filename (e.g., "post_1.txt").
        """
        logger.info(f"Saving as: {filename}")

        full_path = os.path.join(self.config.output_dir, filename)
        if os.path.exists(full_path):
            logger.debug(f"File already exists, will overwrite: {full_path}")

        # Try Ctrl+Shift+S first for explicit "Save As" (works in both old/new Notepad)
        hotkey("ctrl", "shift", "s")
        time.sleep(self.config.save_dialog_wait)

        # Check for save dialog with multiple possible titles
        dialog_found = False
        for title in ("Save As", "Save", "save"):
            if wait_for_window(title, timeout=2.0):
                dialog_found = True
                break

        if not dialog_found:
            # Fallback: try Ctrl+S (opens Save As on first save)
            logger.debug("No Save dialog found, trying Ctrl+S...")
            hotkey("ctrl", "s")
            time.sleep(self.config.save_dialog_wait)
            for title in ("Save As", "Save", "save"):
                if wait_for_window(title, timeout=2.0):
                    dialog_found = True
                    break

        if dialog_found:
            # Navigate to the filename field
            # In Win11 file picker, the filename field is typically already focused
            # Clear existing text and type the full path
            time.sleep(0.3)

            # Type the full path into the filename field via clipboard
            # (clipboard avoids issues with backslashes and special chars)
            hotkey("ctrl", "a")
            time.sleep(0.1)
            type_text(full_path, use_clipboard=True)
            time.sleep(self.config.action_delay)

            # Press Enter to save
            press_key("enter")
            time.sleep(self.config.action_delay)

            # Handle "file already exists" / "Replace" confirmation
            time.sleep(0.5)
            if wait_for_window("Confirm", timeout=1.5):
                hotkey("alt", "y")
                time.sleep(0.3)

            # Some dialogs say "Replace" or "already exists"
            if wait_for_window("Replace", timeout=0.5):
                press_key("enter")
                time.sleep(0.3)
            elif is_window_open("already exists"):
                press_key("enter")
                time.sleep(0.3)
        else:
            logger.warning(
                "Could not find Save dialog. "
                "File may not have been saved to the correct location."
            )

        # Verify the file was actually saved
        time.sleep(0.5)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            logger.info(f"File verified: {full_path} ({file_size} bytes)")
        else:
            logger.warning(f"File NOT found at expected path: {full_path}")

    def _close_notepad(self) -> None:
        """Close the current Notepad window."""
        logger.debug("Closing Notepad...")

        # Try WM_CLOSE first (cleaner)
        if not close_window("Notepad"):
            # Fallback: Alt+F4
            hotkey("alt", "F4")

        time.sleep(self.config.action_delay)

        # Handle "Do you want to save?" dialog if it appears
        if wait_for_window("Notepad", timeout=1.0):
            # If Notepad is still open with a save prompt, click "Don't Save"
            hotkey("alt", "n")  # "Don't Save" in most locales
            time.sleep(0.3)

            # If still open, try pressing the button differently
            if is_window_open("Notepad"):
                press_key("tab")
                press_key("tab")
                press_key("enter")
                time.sleep(0.3)

    def _emergency_cleanup(self) -> None:
        """Attempt to recover from errors by cleaning up Notepad."""
        logger.warning("Emergency cleanup: killing Notepad process...")
        kill_process("Notepad.exe")
        time.sleep(0.5)
        time.sleep(0.5)


def create_demo_screenshots(config: Config | None = None) -> None:
    """Generate the 3 required annotated screenshots for deliverables.

    This function is used during development/demonstration to produce
    screenshots showing icon detection at different positions.
    """
    config = config or Config()
    grounder = VisualGrounder(config)
    grounder.load_model()

    positions = [
        ("top-left", "Icon detected in top-left area"),
        ("bottom-right", "Icon detected in bottom-right area"),
        ("center", "Icon detected in center of screen"),
    ]

    for hint, description in positions:
        logger.info(f"\nCapturing screenshot for: {description}")
        input(f"Move the Notepad icon to the {hint} area and press Enter...")

        screenshot = capture_screenshot()
        detection = grounder.ground(screenshot, config.notepad_icon_label, region_hint=hint)

        path = os.path.join(config.screenshot_dir, f"demo_{hint.replace('-', '_')}.png")

        if detection:
            save_screenshot(
                screenshot,
                path,
                annotation=description,
                bbox=detection.bbox,
                center=detection.center,
            )
            logger.info(f"✓ Saved: {path}")
        else:
            save_screenshot(
                screenshot,
                path,
                annotation=f"FAILED: {description}",
            )
            logger.warning(f"✗ Detection failed for {hint}")
