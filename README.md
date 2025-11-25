# Depth Studio (Depth Anything) — Project README

**Project**: Depth Studio — Depth estimation demo using the "Depth Anything" ONNX model with optional GPU acceleration.

**Location**: `x:/cricketProject/DepthAnything`

---

**Contents**
- Overview
- Key concepts & techniques
- Repository layout
- Quick start (one-click)
- Manual setup (detailed)
- How the GPU/CPU switching works
- `depth.py` — architecture and implementation notes
- UI / Controls explained
- Troubleshooting & common fixes (GPU-related)
- Extending the project

---

**Overview**

This project demonstrates real-time monocular depth estimation using the "Depth Anything v2 small" ONNX model. The app captures a webcam feed, runs depth inference per-frame, visualizes the depth map (PIP) and computes a distance estimate to the user's head using MediaPipe pose landmarks. The project supports GPU acceleration via ONNX Runtime's CUDA Execution Provider when available.

Key goals:
- Make it simple to run on any Windows machine (automated venv + dependency install).
- Use GPU automatically when available and provide clear terminal feedback when installing/updating runtime components.
- Provide a friendly UI for calibration and live feedback.

---

Key concepts & techniques
- ONNX Runtime: Inference engine used to load and run the depth model. `onnxruntime-gpu` is used to enable CUDA execution provider.
- Model preprocessing: Resize, normalize (ImageNet mean/std), convert to CHW and float32.
- Depth map postprocessing: Squeeze, resize to original frame size, normalize for color mapping.
- MediaPipe Pose: Lightweight pose estimator to find a head/face landmark for distance sampling.
- Calibration via ROI: The user can draw a reference ROI and enter the real-world distance for that object. The app uses a proportional constant to translate raw depth values into meters.
- UI: Tkinter for cross-platform GUI (simple controls, PIP preview, switches for GPU/CPU).

---

Repository layout
```
DepthAnything/
├─ depth.py                  # Main application
├─ run_gpu.bat               # Batch wrapper that launches the PS helper
├─ run_gpu.ps1               # PowerShell installer/launcher (creates venv, installs packages, downloads model, sets PATH)
├─ requirements.txt          # Base Python packages
├─ depth_anything_v2_small_f32.onnx  # Model file (ignored by .gitignore)
├─ .gitignore
└─ README.md
```

---

Quick start (one-click)
- Double-click `run_gpu.bat` (Windows). The batch calls `run_gpu.ps1` which:
  - Creates a virtual environment at `./venv` (if missing).
  - Activates the venv and upgrades `pip`.
  - Installs packages from `requirements.txt` (shows live feedback in the terminal).
  - Attempts best-effort installation of CUDA runtime wheel packages if an NVIDIA GPU is detected.
  - Downloads the ONNX model if missing (shows progress in the terminal).
  - Adds any local CUDA wheel `bin` folders to the PATH and launches `depth.py` inside the venv.

If you prefer CLI:
```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\run_gpu.ps1
```

---

Manual setup (detailed)
1. Ensure Python 3.10+ is installed and on PATH.
2. From the project folder run (recommended using PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements.txt
```
3. If you want GPU acceleration, install the matching ONNX Runtime wheel for your CUDA/CUDNN version. The script attempts to install `onnxruntime-gpu` and helper NVIDIA wheels, but on some Windows systems you may still need to install the official NVIDIA CUDA Toolkit and cuDNN:
   - Verify drivers & GPU: `nvidia-smi`
   - Install CUDA 12.x + cuDNN 9.x if required (as indicated by ONNX Runtime errors).
   - Install `onnxruntime-gpu` using pip (choose the version that matches CUDA):
```powershell
pip install onnxruntime-gpu
```
4. Download the model manually if needed:
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" -OutFile depth_anything_v2_small_f32.onnx
```

---

How the GPU/CPU switching works
- `depth.py` contains `switch_processor(target_mode)` which:
  - Ensures the model file exists (downloads it if missing).
  - Attempts to create an ONNX Runtime `InferenceSession` with providers `['CUDAExecutionProvider', 'CPUExecutionProvider']` when `target_mode == 'GPU'`.
  - If `CUDAExecutionProvider` fails to load, the script falls back to CPU and shows a warning with guidance.
- Terminal and a small label in the UI show the currently active provider.
- `run_gpu.ps1` helps by trying to install GPU runtime wheel packages (`nvidia-*` pip wheels) and adding their `bin` directories into PATH if present.

Important: ONNX Runtime loading errors often point to missing DLLs (e.g., `cublasLt64_12.dll`). These are provided by the CUDA Toolkit / cuBLAS / cuDNN — the PowerShell script tries to install wheel versions, but on some systems you must install the official NVIDIA packages.

---

`depth.py` — architecture and implementation notes
- Main class: `UnifiedDepthApp`
  - Initializes Tkinter UI and camera feed
  - Manages an ONNX Runtime session `self.session` and `self.input_name` for inference
  - Calls `switch_processor()` early (defaults to trying GPU first)
  - Uses `init_mediapipe()` to create a MediaPipe Pose instance
  - `video_loop()` captures frames, runs inference (if `self.session` exists), overlays the depth PIP and UI markers, then schedules itself via `root.after(10, self.video_loop)`

- Preprocessing: `preprocess(frame)`
  - Resizes to `INPUT_SIZE` (518)
  - Converts to float32 and scales to [0, 1]
  - Normalizes using ImageNet mean/std
  - Transposes to CHW and adds batch dimension: shape `[1, 3, H, W]`

- Inference & postprocessing:
  - `self.session.run(None, {self.input_name: preprocessed})` returns a depth tensor
  - `depth_map = cv2.resize(np.squeeze(depth_raw), (w, h))`
  - If `show_pip` is enabled, the depth map is normalized and color-mapped (INFERNO) and shown as a bottom-right PIP

- Calibration & distance estimation:
  - The user can draw a reference ROI and enter its known real-world distance (meters).
  - The app computes a proportional constant `k = median_ref_raw * real_dist` and then uses `user_dist = k / user_raw` to estimate distance at the head landmark.

- Head tracking uses the first pose landmark (index 0) as the approximate head/top-of-body location.

- Safety: If inference raises an exception, the loop prints a message and continues (so a failed GPU session won't crash the UI permanently).

---

UI / Controls explained
- Left: Live video feed with optional selection/drawing to set a reference ROI.
  - Click and drag to draw a box when in drawing mode.
- Right: Control panel
  - Hardware Acceleration: Buttons `SWITCH TO GPU` and `SWITCH TO CPU` to force provider change.
  - Camera Settings: Exposure and Gain sliders.
  - Calibration: Enter distance (meters) and `Update` button; `DRAW BOX MODE` to select ROI; `Clear Reference` to clear.
  - Live Data: Shows the raw reference value. Checkbox to toggle the PIP robot vision overlay.

---

Troubleshooting & common fixes (GPU-related)
- Error: `cublasLt64_12.dll missing` or `Failed to create CUDAExecutionProvider`
  - Cause: ONNX Runtime's CUDA provider depends on specific CUDA/CUDNN DLLs not present in PATH.
  - Fixes:
    1. Ensure NVIDIA drivers are installed and up-to-date: `nvidia-smi`
    2. Install the CUDA Toolkit matching the ONNX Runtime build (e.g., CUDA 12.x) and cuDNN 9.x.
    3. Add CUDA `bin` folders to the system PATH (or use the `run_gpu.ps1` automatic additions if the pip wheels installed runtime libs into venv).
    4. Install a matching `onnxruntime-gpu` pip wheel (match ONNX Runtime version to CUDA version).
- If you see `Available providers: ['CPUExecutionProvider']` and GPU fails, the session still runs on CPU — the UI will fall back and continue working.

---

Notes on distribution & Git
- `venv/` and the ONNX model are large and are excluded via `.gitignore`.
- Commit only source files and `requirements.txt`. Reproducible installation is provided by `run_gpu.ps1`.

---

Extending the project
- Swap the ONNX model to a different depth estimator: change `MODEL_URL` and adjust `preprocess()` if the model expects a different input size or normalization.
- Add more sophisticated calibration (multi-point, planar homography) for higher accuracy.
- Replace Tkinter with a more modern UI framework (PyQt, DearPyGui) if you want richer controls and better cross-platform visuals.

---

Credits & license
- The Depth Anything ONNX model is from `onnx-community` on Hugging Face.
- MediaPipe is used for pose detection.
- This repository contains glue code and UI scaffolding. Check licenses of upstream components before redistributing.

---

If you want, I can also add a short `USAGE.md` with screenshots and sample outputs, or create a small test harness to measure FPS and inference times. Which would you prefer next?
