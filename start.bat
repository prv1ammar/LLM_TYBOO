@echo off
REM Quick start script for Windows (WSL2 required)

echo 🚀 Self-Hosted AI Stack - Quick Start (Windows)
echo ==================================================
echo.

REM Check if WSL is installed
wsl --list >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ WSL is not installed. Please install WSL2 first.
    echo    Visit: https://docs.microsoft.com/en-us/windows/wsl/install
    exit /b 1
)

echo ✅ WSL detected
echo.
echo 📝 This script will launch the setup in WSL2
echo    Make sure you have Docker Desktop with WSL2 backend enabled
echo.
pause

REM Convert Windows path to WSL path
set "SCRIPT_DIR=%~dp0"
set "WSL_PATH=/mnt/c/Users/info/Desktop/llm"

echo 🐧 Launching in WSL...
wsl -d Ubuntu bash -c "cd %WSL_PATH% && bash start.sh"

echo.
echo ✅ Setup complete!
echo.
pause
