# Running the Vision-Based Desktop Automation

To run this application, which uses Florence-2 to ground desktop elements, follow these steps:

## 1. Prerequisites
- **OS**: Windows 10/11
- **Python**: 3.10 or newer (found at `C:\Users\Elbon\AppData\Local\Programs\Python\Python313\python.exe`)
- **Desktop Setup**: Ensure you have a **Notepad** shortcut on your desktop. The automation specifically looks for this to demonstrate its capabilities.

## 2. Environment Setup

It is highly recommended to use a virtual environment. Open your terminal in the project root and run these as **two separate commands**:

```powershell
# 1. Create a virtual environment using the full path to Python
& "C:\Users\Elbon\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Install the project in editable mode with development dependencies:

```powershell
pip install -e ".[dev]"
```

> [!NOTE]
> If you have an NVIDIA GPU, it is recommended to install PyTorch with CUDA for significantly faster detection:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## 4. Running the Application

Once installed, you can use the `vision-auto` command or run it as a module.

### Full Automation Run
This will process 10 posts, save them to your desktop, and use vision to locate the Notepad icon.
```powershell
vision-auto run
```

### Try Detection (Grounding)
You can test the grounding engine on any desktop element:
```powershell
vision-auto ground "Notepad"
# or
vision-auto ground "Recycle Bin"
```

### Run as Module
If the CLI command is not in your path yet:
```powershell
& "C:\Users\Elbon\AppData\Local\Programs\Python\Python313\python.exe" -m vision_desktop_automation run
```

## 5. Troubleshooting
- **Icon not found**: Ensure the Notepad icon is visible on the primary monitor.
- **CPU/GPU**: By default, it tries to use `cuda`. Use `--device cpu` if you don't have a supported GPU:
  `vision-auto run --device cpu`
