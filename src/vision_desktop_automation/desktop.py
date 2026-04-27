"""Desktop interaction utilities — mouse, keyboard, window management."""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

try:
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
except ImportError:
    pyautogui = None

try:
    import win32gui
    import win32con
    import win32process
    import psutil
except ImportError:
    win32gui = None
    win32con = None
    win32process = None
    psutil = None


def double_click(x: int, y: int) -> None:
    """Double-click at the given screen coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
    """
    logger.debug(f"Double-clicking at ({x}, {y})")
    pyautogui.moveTo(x, y, duration=0.3)
    time.sleep(0.1)
    pyautogui.doubleClick(x, y)


def single_click(x: int, y: int) -> None:
    """Single-click at the given screen coordinates."""
    logger.debug(f"Clicking at ({x}, {y})")
    pyautogui.click(x, y)


def type_text(text: str, interval: float = 0.02, use_clipboard: bool = False) -> None:
    """Type text using the keyboard.

    For short, single-line, ASCII-only text: uses pyautogui.write().
    For multiline text, long text, or when use_clipboard=True: uses
    clipboard paste (Ctrl+V) which is faster and handles all characters.

    Args:
        text: The text to type.
        interval: Delay between keystrokes (seconds).
        use_clipboard: Force clipboard paste mode.
    """
    logger.debug(f"Typing {len(text)} characters (clipboard={use_clipboard})")

    # Use clipboard for multiline, long text, or when forced
    if use_clipboard or "\n" in text or len(text) > 200:
        _type_via_clipboard(text)
    else:
        try:
            text.encode("ascii")
            pyautogui.write(text, interval=interval)
        except UnicodeEncodeError:
            _type_via_clipboard(text)


def _type_via_clipboard(text: str) -> None:
    """Type text by copying to clipboard and pasting.

    Uses pyperclip for reliable cross-platform clipboard access.
    Falls back to PowerShell Set-Clipboard if pyperclip isn't available.
    """
    try:
        import pyperclip
        pyperclip.copy(text)
    except ImportError:
        # Fallback: use PowerShell to set clipboard
        import subprocess
        # PowerShell Set-Clipboard handles all text correctly
        subprocess.run(
            ["powershell", "-Command", f"Set-Clipboard -Value $input"],
            input=text,
            capture_output=True,
            text=True,
            timeout=5,
        )

    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.3)


def hotkey(*keys: str) -> None:
    """Press a keyboard hotkey combination.

    Args:
        keys: Key names (e.g., "ctrl", "s").
    """
    logger.debug(f"Hotkey: {'+'.join(keys)}")
    pyautogui.hotkey(*keys)


def press_key(key: str) -> None:
    """Press a single key."""
    logger.debug(f"Press key: {key}")
    pyautogui.press(key)


def wait_for_window(
    title_contains: str,
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for a window with the given title to appear.

    Args:
        title_contains: Substring to search for in window titles.
        timeout: Maximum wait time in seconds.
        poll_interval: Time between checks in seconds.

    Returns:
        True if the window was found, False if timed out.
    """
    if win32gui is None:
        logger.warning("win32gui not available, using simple delay fallback")
        time.sleep(timeout / 2)
        return True

    logger.debug(f"Waiting for window containing '{title_contains}'...")
    start = time.time()

    while time.time() - start < timeout:
        hwnd = _find_window(title_contains)
        if hwnd:
            logger.info(f"Found window: '{title_contains}' (hwnd={hwnd})")
            return True
        time.sleep(poll_interval)

    logger.warning(f"Window '{title_contains}' not found after {timeout}s")
    return False


def _find_window(title_contains: str) -> Optional[int]:
    """Find a window handle by partial title match or class name."""
    result = []
    search = title_contains.lower()

    def enum_callback(hwnd: int, _: None) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if search in title.lower():
                result.append(hwnd)
                return True
            # Also check window class name (Win11 Notepad uses class "Notepad")
            try:
                class_name = win32gui.GetClassName(hwnd)
                if search in class_name.lower():
                    result.append(hwnd)
                    return True
            except Exception:
                pass
            # Check by process name as a fallback
            if win32process is not None and psutil is not None:
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    proc = psutil.Process(pid)
                    if search in proc.name().lower():
                        result.append(hwnd)
                except Exception:
                    pass
        return True

    win32gui.EnumWindows(enum_callback, None)
    return result[0] if result else None


def focus_window(title_contains: str) -> bool:
    """Bring a window to the foreground.

    Args:
        title_contains: Substring to search for in window titles.

    Returns:
        True if window was focused, False otherwise.
    """
    if win32gui is None:
        return False

    hwnd = _find_window(title_contains)
    if hwnd:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.3)
            logger.debug(f"Focused window: hwnd={hwnd}")
            return True
        except Exception as e:
            logger.warning(f"Could not focus window: {e}")
    return False


def close_window(title_contains: str) -> bool:
    """Close a window by sending WM_CLOSE.

    Args:
        title_contains: Substring to search for in window titles.

    Returns:
        True if a close message was sent, False otherwise.
    """
    if win32gui is None:
        # Fallback: Alt+F4
        pyautogui.hotkey("alt", "F4")
        time.sleep(0.5)
        return True

    hwnd = _find_window(title_contains)
    if hwnd:
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        logger.debug(f"Sent WM_CLOSE to hwnd={hwnd}")
        time.sleep(0.5)
        return True

    logger.warning(f"No window found matching '{title_contains}' to close")
    return False


def kill_process(process_name: str) -> None:
    """Forcefully kill all processes with the given name.

    Args:
        process_name: Name of the process (e.g., "notepad.exe").
    """
    if psutil is None:
        import subprocess
        logger.debug(f"psutil not available, killing {process_name} via taskkill")
        subprocess.run(["taskkill", "/f", "/im", process_name], capture_output=True)
        return

    logger.debug(f"Killing all instances of {process_name}")
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and process_name.lower() in proc.info["name"].lower():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def is_window_open(title_contains: str) -> bool:
    """Check if a window with the given title is currently open."""
    if win32gui is None:
        return False
    return _find_window(title_contains) is not None


def minimize_all_windows() -> None:
    """Minimize all windows to show the desktop.

    Uses multiple strategies for reliability:
    1. Shell.Application COM object (most reliable on Windows)
    2. Win+M fallback
    3. Verify by checking remaining foreground windows
    """
    # Strategy 1: Use Shell COM object to toggle desktop
    try:
        import subprocess
        # shell.application MinimizeAll is the most reliable programmatic way
        subprocess.run(
            [
                "powershell", "-Command",
                "(New-Object -ComObject Shell.Application).MinimizeAll()"
            ],
            capture_output=True,
            timeout=5,
        )
        logger.debug("Minimized all windows via Shell.Application COM")
        time.sleep(1.5)
    except Exception as e:
        logger.debug(f"Shell.Application minimize failed: {e}, trying Win+M")
        pyautogui.hotkey("win", "m")
        time.sleep(1.5)

    # Verify: check if non-desktop windows are still in foreground
    if win32gui is not None:
        fg = win32gui.GetForegroundWindow()
        fg_class = ""
        try:
            fg_class = win32gui.GetClassName(fg)
        except Exception:
            pass
        # "Progman" and "WorkerW" are the desktop window classes
        if fg_class not in ("Progman", "WorkerW", "Shell_TrayWnd", ""):
            logger.debug(
                f"Foreground is still '{fg_class}', sending Win+D as last resort"
            )
            pyautogui.hotkey("win", "d")
            time.sleep(1.5)

    # Click on an empty spot in the desktop to ensure it has focus
    # Use a spot near the center-top which is typically empty
    pyautogui.click(960, 10)
    time.sleep(0.3)


def list_visible_windows() -> list[str]:
    """List all visible window titles (useful for debugging).

    Returns:
        List of window title strings.
    """
    if win32gui is None:
        return []

    titles = []

    def enum_callback(hwnd: int, _: None) -> bool:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip():
                try:
                    class_name = win32gui.GetClassName(hwnd)
                    titles.append(f"{title} [class={class_name}]")
                except Exception:
                    titles.append(title)
        return True

    win32gui.EnumWindows(enum_callback, None)
    return titles


def dismiss_popups(timeout: float = 2.0) -> None:
    """Attempt to dismiss unexpected pop-ups by pressing Escape and Enter.

    This is a basic heuristic; the VLM grounding approach can also be used
    to detect and dismiss specific pop-ups dynamically.
    """
    logger.debug("Checking for unexpected pop-ups...")
    time.sleep(0.3)
    # Press Escape to dismiss most dialogs
    pyautogui.press("escape")
    time.sleep(0.5)
