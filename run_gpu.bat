@echo off
REM Wrapper: Runs a PowerShell helper that creates/activates a venv,
REM installs dependencies, downloads the model (if missing), and launches the app.

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_gpu.ps1"

pause
