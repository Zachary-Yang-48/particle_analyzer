#!/usr/bin/env python3
"""
Particle Diameter Analyzer
==========================
Identifies and measures circular particles in microscopy images.

Usage:
    python particle_analyzer.py <image_path> <scale_bar_um> [scale_bar_pixels]

Examples:
    python particle_analyzer.py sample.bmp 100
        # Auto-detect scale bar pixels for 100 μm

    python particle_analyzer.py sample.bmp 100 150
        # Manually specify: 100 μm = 150 pixels

Outputs:
    - <image_name>_analyzed.png : Annotated image with detected particles
    - <image_name>_measurements.csv : CSV with all particle measurements
"""

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import argparse
import os
from pathlib import Path


def detect_scale_bar(gray_image, search_regions=None):
    """
    Attempt to auto-detect the scale bar in the image.
    
    Looks for horizontal lines in corners of the image.
    
    Args:
        gray_image: Grayscale image
        search_regions: List of (y1, y2, x1, x2) tuples for regions to search
                       Default: all four corners
    
    Returns:
        dict with 'length_pixels', 'x1', 'y1', 'x2', 'y2', 'roi'
        or None if not found
    """
    h, w = gray_image.shape
    
    if search_regions is None:
        # Search all four corners
        corner_size_y = min(200, h // 4)
        corner_size_x = min(400, w // 3)
        search_regions = [
            (0, corner_size_y, 0, corner_size_x),                    # Top-left
            (0, corner_size_y, w - corner_size_x, w),                # Top-right
            (h - corner_size_y, h, 0, corner_size_x),                # Bottom-left
            (h - corner_size_y, h, w - corner_size_x, w),            # Bottom-right
        ]
    
    best_result = None
    best_length = 0
    
    for y1, y2, x1, x2 in search_regions:
        roi = gray_image[y1:y2, x1:x2]
        
        # Edge detection
        edges = cv2.Canny(roi, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=5
        )
        
        if lines is None:
            continue
        
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            
            # Check if horizontal (within 3 degrees)
            angle = np.abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180 / np.pi)
            if not (angle < 3 or angle > 177):
                continue
            
            length = np.sqrt((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2)
            
            # Scale bars are typically 50-300 pixels
            if length < 40 or length > 400:
                continue
            
            if length > best_length:
                best_length = length
                best_result = {
                    'length_pixels': round(length),
                    'x1': x1 + lx1,
                    'y1': y1 + ly1,
                    'x2': x1 + lx2,
                    'y2': y1 + ly2,
                    'region': (y1, y2, x1, x2),
                    'roi': roi.copy()
                }
    
    return best_result


def visualize_scale_bar_detection(image, detection_result, output_path=None):
    """
    Create visualization of detected scale bar.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    if detection_result:
        x1, y1 = int(detection_result['x1']), int(detection_result['y1'])
        x2, y2 = int(detection_result['x2']), int(detection_result['y2'])
        
        # Draw detected line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add length annotation
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(vis, f"{detection_result['length_pixels']:.0f}px",
                    (mid_x - 30, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if output_path:
        cv2.imwrite(str(output_path), vis)
    
    return vis


def extract_scale_bar_crop(image, detection_result, padding=20):
    """
    Extract a cropped region around the detected scale bar for manual verification.
    """
    if detection_result is None:
        return None
    
    x1, y1 = int(detection_result['x1']), int(detection_result['y1'])
    x2, y2 = int(detection_result['x2']), int(detection_result['y2'])
    
    h, w = image.shape[:2]
    
    crop_x1 = max(0, min(x1, x2) - padding)
    crop_x2 = min(w, max(x1, x2) + padding)
    crop_y1 = max(0, min(y1, y2) - padding * 2)
    crop_y2 = min(h, max(y1, y2) + padding * 2)
    
    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def detect_particles(gray_image):
    """
    Detect particles using adaptive threshold + watershed segmentation.
    
    Returns list of particle dictionaries with measurements.
    """
    h, w = gray_image.shape
    
    # Adaptive thresholding
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=21, C=6
    )
    
    # Fill holes to handle hollow particles
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = adaptive.copy()
    cv2.floodFill(filled, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    filled_particles = adaptive | filled_inv
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    filled_particles = cv2.morphologyEx(filled_particles, cv2.MORPH_OPEN, kernel, iterations=2)
    filled_particles = cv2.morphologyEx(filled_particles, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Distance transform for watershed
    dist_transform = ndimage.distance_transform_edt(filled_particles)
    
    # Find local maxima (particle centers)
    coords = peak_local_max(
        dist_transform,
        min_distance=4,
        threshold_rel=0.15,
        labels=filled_particles,
        exclude_border=False
    )
    
    # Create markers for watershed
    markers = np.zeros(filled_particles.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    markers = ndimage.grey_dilation(markers, size=(3, 3))
    
    # Watershed segmentation
    labels = watershed(-dist_transform, markers, mask=filled_particles)
    
    # First pass: collect particles with good circularity to establish baseline
    good_circ_particles = []
    for label_id in range(1, labels.max() + 1):
        mask_single = (labels == label_id).astype(np.uint8)
        area = np.sum(mask_single)
        
        if area < 50:
            continue
        
        contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        cnt = contours[0]
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if circularity > 0.75:
            good_circ_particles.append({'area': area, 'radius': radius, 'circularity': circularity})
    
    # Calculate size statistics
    median_area = np.median([p['area'] for p in good_circ_particles]) if good_circ_particles else 400
    q75_area = np.percentile([p['area'] for p in good_circ_particles], 75) if good_circ_particles else 600
    max_allowed_area = q75_area * 2.5
    
    # Second pass: filter with size awareness
    particles = []
    for label_id in range(1, labels.max() + 1):
        mask_single = (labels == label_id).astype(np.uint8)
        area = np.sum(mask_single)
        
        if area < 50 or area > max_allowed_area:
            continue
        
        contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        cnt = contours[0]
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        enclosing_area = np.pi * radius ** 2
        fill_ratio = area / enclosing_area if enclosing_area > 0 else 0
        
        # Size-dependent circularity threshold
        if area > q75_area * 1.5:
            min_circ = 0.65
        elif area > median_area * 1.5:
            min_circ = 0.55
        else:
            min_circ = 0.45
        
        if circularity < min_circ:
            continue
        
        # Hollow detection: compare center brightness vs ring brightness
        cx_int, cy_int = int(cx), int(cy)
        r_center = max(1, int(radius * 0.25))
        y1c, y2c = max(0, cy_int - r_center), min(h, cy_int + r_center + 1)
        x1c, x2c = max(0, cx_int - r_center), min(w, cx_int + r_center + 1)
        center_val = np.mean(gray_image[y1c:y2c, x1c:x2c]) if (y2c > y1c and x2c > x1c) else 0
        
        ring_vals = []
        r_int = max(3, int(radius))
        for angle in range(0, 360, 30):
            for r_frac in [0.6, 0.75]:
                rx = int(cx + r_int * r_frac * np.cos(np.radians(angle)))
                ry = int(cy + r_int * r_frac * np.sin(np.radians(angle)))
                if 0 <= rx < w and 0 <= ry < h:
                    ring_vals.append(gray_image[ry, rx])
        ring_val = np.mean(ring_vals) if ring_vals else center_val
        is_hollow = (center_val - ring_val) > 15
        
        particles.append({
            'center_x': cx,
            'center_y': cy,
            'radius': radius,
            'diameter_px': 2 * radius,
            'area_px': area,
            'circularity': circularity,
            'fill_ratio': fill_ratio,
            'is_hollow': is_hollow
        })
    
    return particles


def create_annotated_image(original_image, particles):
    """
    Create annotated image with detected particles marked.
    
    Colors:
        - Yellow: Hollow particles
        - Green: Filled particles
        - Red dot: Center point
        - Blue text: Particle ID
    """
    vis = original_image.copy()
    
    for i, p in enumerate(particles):
        cx, cy = int(p['center_x']), int(p['center_y'])
        r = int(p['radius'])
        particle_id = i + 1
        
        # Circle color based on hollow/filled
        color = (0, 255, 255) if p['is_hollow'] else (0, 255, 0)  # Yellow or Green (BGR)
        
        # Draw circle and center
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)  # Red center
        
        # Add ID label
        cv2.putText(vis, str(particle_id), (cx + r + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)  # Blue text
    
    return vis


def create_measurements_csv(particles, um_per_pixel):
    """
    Create DataFrame with particle measurements.
    """
    records = []
    for i, p in enumerate(particles):
        records.append({
            'particle_id': i + 1,
            'center_x_px': round(p['center_x'], 1),
            'center_y_px': round(p['center_y'], 1),
            'diameter_px': round(p['diameter_px'], 2),
            'diameter_um': round(p['diameter_px'] * um_per_pixel, 2),
            'area_px': int(p['area_px']),
            'circularity': round(p['circularity'], 3),
            'is_hollow': p['is_hollow']
        })
    
    return pd.DataFrame(records)


def analyze_image(image_path, scale_bar_um, scale_bar_pixels=None, output_dir=None, non_interactive=False):
    """
    Main analysis function.
    
    Args:
        image_path: Path to input image
        scale_bar_um: Scale bar length in micrometers
        scale_bar_pixels: Scale bar length in pixels (auto-detect if None)
        output_dir: Output directory (default: same as input)
        non_interactive: If True, skip confirmation prompts
    
    Returns:
        DataFrame with measurements
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Setup output directory
    input_path = Path(image_path)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path('.')  # Current directory as fallback
    
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")
    
    # Auto-detect scale bar if not provided
    if scale_bar_pixels is None:
        print("\nAuto-detecting scale bar...")
        detection = detect_scale_bar(gray)
        
        if detection:
            scale_bar_pixels = detection['length_pixels']
            print(f"  Detected scale bar: {scale_bar_pixels:.0f} pixels")
            print(f"  Location: ({detection['x1']:.0f}, {detection['y1']:.0f}) to ({detection['x2']:.0f}, {detection['y2']:.0f})")
            
            # Save scale bar visualization
            scale_vis_path = out_dir / f"{input_path.stem}_scale_bar.png"
            visualize_scale_bar_detection(img, detection, scale_vis_path)
            print(f"  Saved: {scale_vis_path}")
            
            # Save cropped scale bar region
            crop = extract_scale_bar_crop(img, detection)
            if crop is not None:
                crop_path = out_dir / f"{input_path.stem}_scale_bar_crop.png"
                cv2.imwrite(str(crop_path), crop)
                print(f"  Saved crop: {crop_path}")
            
            if not non_interactive:
                # Ask for confirmation
                print(f"\n  Detected: {scale_bar_pixels:.0f} pixels = {scale_bar_um} μm")
                response = input("  Is this correct? [Y/n/enter new value]: ").strip()
                
                if response.lower() == 'n':
                    scale_bar_pixels = float(input("  Enter correct scale bar length in pixels: "))
                elif response and response.lower() != 'y':
                    try:
                        scale_bar_pixels = float(response)
                    except ValueError:
                        pass  # Keep detected value
        else:
            print("  Could not auto-detect scale bar.")
            if non_interactive:
                raise ValueError("Scale bar auto-detection failed. Please provide scale_bar_pixels manually.")
            scale_bar_pixels = float(input("  Enter scale bar length in pixels: "))
    
    # Calculate scale
    um_per_pixel = scale_bar_um / scale_bar_pixels
    
    print(f"\nScale: {um_per_pixel:.4f} μm/pixel ({scale_bar_um} μm = {scale_bar_pixels:.0f} px)")
    
    # Detect particles
    print("\nDetecting particles...")
    particles = detect_particles(gray)
    print(f"Detected {len(particles)} particles")
    
    # Statistics
    hollow_count = sum(1 for p in particles if p['is_hollow'])
    diameters_um = [p['diameter_px'] * um_per_pixel for p in particles]
    
    print(f"\nStatistics:")
    print(f"  Hollow particles: {hollow_count}")
    print(f"  Filled particles: {len(particles) - hollow_count}")
    print(f"  Diameter range: {min(diameters_um):.1f} - {max(diameters_um):.1f} μm")
    print(f"  Mean diameter: {np.mean(diameters_um):.1f} μm")
    print(f"  Median diameter: {np.median(diameters_um):.1f} μm")
    
    # Create outputs
    input_path = Path(image_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_path.stem
    
    # Annotated image
    annotated = create_annotated_image(img, particles)
    annotated_path = output_dir / f"{base_name}_analyzed.png"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"\nSaved annotated image: {annotated_path}")
    
    # CSV
    df = create_measurements_csv(particles, um_per_pixel)
    csv_path = output_dir / f"{base_name}_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved measurements: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze particle diameters in microscopy images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python particle_analyzer.py image.bmp 100
        Analyze image.bmp with 100 μm scale bar (auto-detect pixels)

    python particle_analyzer.py image.bmp 100 150
        Analyze image.bmp where the 100 μm scale bar is 150 pixels

    python particle_analyzer.py image.bmp 50 --output ./results
        Save outputs to ./results directory
        
    python particle_analyzer.py image.bmp 100 150 --non-interactive
        Run without confirmation prompts (for batch processing)
        """
    )
    
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('scale_bar_um', type=float, help='Scale bar length in micrometers')
    parser.add_argument('scale_bar_pixels', type=float, nargs='?', default=None,
                        help='Scale bar length in pixels (auto-detect if not provided)')
    parser.add_argument('--output', '-o', help='Output directory (default: current directory)')
    parser.add_argument('--non-interactive', '-n', action='store_true',
                        help='Skip confirmation prompts (for batch processing)')
    
    args = parser.parse_args()
    
    analyze_image(args.image_path, args.scale_bar_um, args.scale_bar_pixels, 
                  args.output, args.non_interactive)


if __name__ == '__main__':
    main()