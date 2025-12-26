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
echo ========================================
echo IMPORTANT: Look for the URL below!
echo ========================================
echo.
echo The application should open automatically in your browser.
echo If it doesn't, look for a line that says:
echo   "You can now view your Streamlit app in your browser."
echo.
echo The URL will be something like:
echo   http://localhost:8501
echo.
echo Copy that URL and paste it into your web browser.
echo.
echo Press Ctrl+C to stop the application.
echo.
echo ========================================
echo.

streamlit run gui_streamlit.py

pause

