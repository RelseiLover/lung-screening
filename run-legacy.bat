@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
"%PYTHON_EXE%" run.py --legacy
if errorlevel 1 pause
