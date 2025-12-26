@echo off
REM Launch script for Particle Analyzer Streamlit GUI
REM This script starts the Streamlit web interface

echo Starting Particle Analyzer GUI...
echo.

REM Activate virtual environment (created with uv)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo ERROR: Virtual environment not found!
    echo Please run setup first:
    echo   uv venv
    echo   uv pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the application.
echo.

streamlit run gui_streamlit.py

pause

