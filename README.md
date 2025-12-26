# Particle Analyzer

A tool for identifying and measuring circular particles in microscopy images with batch processing capabilities.

## Features

- 🔬 Automatic particle detection using advanced image processing
- 📏 Scale bar auto-detection or manual specification
- 📊 Batch processing of multiple images
- 📁 Excel output with multiple sheets (images + measurements)
- 🖥️ Simple web-based GUI for non-technical users

## Installation

### Prerequisites

1. Install Python 3.7 or higher
2. Install [uv](https://github.com/astral-sh/uv) (fast Python package installer):
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Mac/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Setup

**Windows users (recommended):**
1. Double-click `setup_env.bat` to automatically create the environment and install dependencies
2. Then double-click `launch_gui.bat` to start the application

**Manual setup (all platforms):**
1. Create virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

   Or in one command:
   ```bash
   uv venv && uv pip install -r requirements.txt
   ```

2. **Windows users**: Double-click `launch_gui.bat`

3. **Mac/Linux users**: Activate the environment and run:
   ```bash
   source .venv/bin/activate
   streamlit run gui_streamlit.py
   ```

## Usage

### GUI Mode (Recommended for non-technical users)

1. **Windows**: Double-click `launch_gui.bat`
2. **Mac/Linux**: Run `streamlit run gui_streamlit.py`
3. The web interface will open in your browser
4. Set input/output directories and scale bar settings
5. Click "Process All Images" to batch process

### Command Line Mode

```bash
# Auto-detect scale bar
python analyzer.py image.bmp 100

# Manual scale bar specification
python analyzer.py image.bmp 100 150

# Batch processing with output directory
python analyzer.py image.bmp 100 --output ./results --non-interactive
```

## Output Format

Each processed image generates an Excel file (`.xlsx`) with the following sheets:

- **image**: Annotated image with detected particles (colored circles)
- **scale_bar**: Scale bar visualization (if auto-detected)
- **scale_bar_crop**: Cropped scale bar region (if auto-detected)
- **measurements**: Particle measurement data (CSV format)

## Supported Image Formats

- BMP
- PNG
- JPG/JPEG
- TIF/TIFF

## Requirements

See `requirements.txt` for full list of dependencies.

## Notes

- Scale bars should be visible in the image corners for best auto-detection
- Images are processed sequentially in batch mode
- All outputs are saved to the specified output directory

