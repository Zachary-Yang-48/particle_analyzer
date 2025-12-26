@echo off
REM Setup script for Particle Analyzer
REM Creates virtual environment and installs dependencies using uv

echo ========================================
echo Particle Analyzer - Environment Setup
echo ========================================
echo.

REM Check if uv is installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: uv is not installed or not in PATH!
    echo.
    echo Please install uv first:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    uv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)

echo.
echo Step 2: Installing required packages...
call .venv\Scripts\activate.bat
uv pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install packages!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo You can now run launch_gui.bat to start the application.
echo.
pause

