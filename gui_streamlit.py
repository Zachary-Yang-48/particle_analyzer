#!/usr/bin/env python3
"""
Streamlit GUI for Particle Analyzer - Batch Processor
======================================================
Simple web-based interface for batch processing microscopy images.
"""

import streamlit as st
from pathlib import Path
import glob
from analyzer import analyze_image
import traceback

st.set_page_config(
    page_title="Particle Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Particle Analyzer - Batch Processor")
st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Directory selection
    st.subheader("Directories")
    input_dir = st.text_input(
        "Input Directory",
        value="./images",
        help="Directory containing images to process"
    )
    
    output_dir = st.text_input(
        "Output Directory",
        value="./results",
        help="Directory where results will be saved"
    )
    
    # Scale bar settings
    st.subheader("Scale Bar Configuration")
    scale_bar_um = st.number_input(
        "Scale Bar Length (μm)",
        min_value=0.1,
        value=100.0,
        step=1.0,
        help="Physical length of the scale bar in micrometers"
    )
    
    auto_detect = st.checkbox(
        "Auto-detect scale bar pixels",
        value=True,
        help="Automatically detect the scale bar length in pixels. Uncheck to specify manually."
    )
    
    scale_bar_pixels = None
    if not auto_detect:
        scale_bar_pixels = st.number_input(
            "Scale Bar Pixels",
            min_value=1,
            value=150,
            step=1,
            help="Length of the scale bar in pixels"
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Image Processing")
    
    # Check if directories exist
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        st.warning(f"⚠️ Input directory does not exist: {input_dir}")
        st.info("Please enter a valid input directory path.")
    else:
        # Find image files
        image_extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        image_files = sorted(set(image_files))  # Remove duplicates and sort
        
        if not image_files:
            st.warning(f"⚠️ No image files found in: {input_dir}")
            st.info("Supported formats: BMP, PNG, JPG, JPEG, TIF, TIFF")
        else:
            st.success(f"✓ Found {len(image_files)} image(s) to process")
            
            # Display file list
            with st.expander(f"📋 View {len(image_files)} image file(s)"):
                for img_file in image_files:
                    st.text(f"  • {img_file.name}")
            
            # Process button
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                if not scale_bar_um or scale_bar_um <= 0:
                    st.error("❌ Please enter a valid scale bar length (μm)")
                else:
                    # Create output directory
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Process images
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    success_count = 0
                    error_count = 0
                    results = []
                    
                    for i, img_path in enumerate(image_files):
                        try:
                            status_text.text(f"Processing: {img_path.name} ({i+1}/{len(image_files)})")
                            
                            # Process the image
                            df = analyze_image(
                                str(img_path),
                                scale_bar_um,
                                scale_bar_pixels,
                                str(output_path),
                                non_interactive=True
                            )
                            
                            particle_count = len(df)
                            results.append({
                                'file': img_path.name,
                                'status': '✓ Success',
                                'particles': particle_count
                            })
                            success_count += 1
                            
                        except Exception as e:
                            error_msg = str(e)
                            results.append({
                                'file': img_path.name,
                                'status': f'✗ Error: {error_msg[:50]}...',
                                'particles': 0
                            })
                            error_count += 1
                            st.error(f"Error processing {img_path.name}: {error_msg}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(image_files))
                    
                    # Final status
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display results summary
                    st.markdown("---")
                    st.header("📊 Processing Results")
                    
                    col_success, col_error, col_total = st.columns(3)
                    with col_success:
                        st.metric("✅ Successful", success_count)
                    with col_error:
                        st.metric("❌ Errors", error_count)
                    with col_total:
                        st.metric("📁 Total Files", len(image_files))
                    
                    # Display detailed results
                    with results_container:
                        st.subheader("📋 Detailed Results")
                        for result in results:
                            if '✓' in result['status']:
                                st.success(f"{result['status']} - **{result['file']}** ({result['particles']} particles detected)")
                            else:
                                st.error(f"{result['status']} - **{result['file']}**")
                    
                    if success_count > 0:
                        st.success(f"🎉 Processing complete! Results saved to: {output_path}")
                        st.info(f"💡 Each processed image has an Excel file (.xlsx) with multiple sheets containing the annotated image and measurements.")

with col2:
    st.header("ℹ️ Instructions")
    st.markdown("""
    ### How to use:
    
    1. **Set Input Directory**
       - Enter the folder path containing your microscopy images
    
    2. **Set Output Directory**
       - Enter where you want results saved
    
    3. **Configure Scale Bar**
       - Enter the physical length (μm) of your scale bar
       - Enable auto-detection or specify pixels manually
    
    4. **Process Images**
       - Click "Process All Images" to start batch processing
    
    ### Output:
    Each processed image generates an Excel file (`.xlsx`) with:
    - **image** sheet: Annotated image with detected particles
    - **scale_bar** sheet: Scale bar visualization (if auto-detected)
    - **scale_bar_crop** sheet: Scale bar crop region (if auto-detected)
    - **measurements** sheet: Particle measurement data
    
    ### Supported Formats:
    - BMP, PNG, JPG, JPEG, TIF, TIFF
    """)
    
    st.markdown("---")
    st.markdown("**💡 Tip:** Make sure your images have visible scale bars for best results!")

