# Metashape Spherical to Cubemap Converter v012

A Python script for Agisoft Metashape Pro that converts a set of spherical (equirectangular) images into separate images for the faces of a cubemap projection (front, back, left, right, top, bottom). It creates new cameras in Metashape for each cube face, inheriting the position and orientation of the original spherical camera.

Version 012 includes stability and memory management improvements, especially when dealing with a large number of cameras.

## ⚠️ CRITICAL UPDATE: Fixed Version Available

**IMPORTANT:** Critical mathematical and geometric errors have been discovered and fixed in the original v012 script. It is strongly recommended to use the **new fixed script `unified_fixed_v002.py`** instead of previous versions.

### 🎯 What's Fixed in unified_fixed_v002.py:

✅ **Projection Mathematics**: Completely rewritten equirectangular → cubemap conversion function  
✅ **Camera Positions**: All cubemap cameras now correctly positioned at the spherical camera center  
✅ **Face Orientations**: Fixed directions for all 6 cube faces (front, back, left, right, top, down)  
✅ **Color Extraction**: Improved algorithm for extracting colored sparse point cloud from Metashape  
✅ **COLMAP Export**: Automatic creation of COLMAP structure folder for 3D Gaussian Splatting

### 🚀 New Simplified Workflow:

1. **Import spherical images** into Metashape
2. **Align cameras** - **once only**
3. **Run unified_fixed_v002.py**
4. ✅ **Done!** - COLMAP export folder created automatically

**❗ IMPORTANT**: After running the fixed script, **NO NEED** to realign cameras in Metashape. The script creates correctly oriented cubemap cameras AND exports ready COLMAP structure.

### 📁 Export Structure:

```
output_folder/
├── images/           # Cubemap faces with correct geometry
├── sparse/0/         # COLMAP data for 3DGS
│   ├── cameras.bin   # Camera parameters
│   ├── images.bin    # Positions and orientations (fixed!)
│   └── points3D.bin  # Colored sparse point cloud
└── README_FIXED.txt  # Detailed documentation
```

### 🔄 Update Status:

- **Current version**: `unified_fixed_v002.py` - quick critical fix
- **Planned**: Full v013 update integrating all fixes into the main GUI script
- **Recommendation**: Use `unified_fixed_v002.py` for all new projects

## Features

- **Graphical User Interface (GUI)**: User-friendly interface based on PyQt5 for easy parameter configuration and progress monitoring.
- **Console Mode**: Ability to run without a GUI, prompting for parameters via standard Metashape dialogs.
- **Multithreading**: Parallel processing to speed up conversion:
  - Parallel processing of faces for a _single_ camera.
  - Parallel processing of _multiple_ cameras simultaneously (configurable).
- **Face Selection**: Option to generate only specific cube faces.
- **Parameter Customization**: Control overlap, face size, file format (JPG, PNG, TIFF), compression quality, interpolation method.
- **Coordinate System Detection**: Automatic detection of the project's primary orientation (Y-Up, Z-Up, X-Up) for correct camera creation.
- **Post-processing Options**: Optional realignment of cameras and removal of original spherical cameras after conversion.
- **Cyrillic Path Support**: Correct handling of file paths containing non-Latin characters.
- **Automatic Dependency Installation**: Checks and attempts to install missing `opencv-python` and `PyQt5` libraries.
- **Multilingual**: Interface available in English and Russian (automatically detected based on Metashape settings).

## Requirements

- **Agisoft Metashape Pro**: Version 1.6 or newer recommended.
- **Python**: Version 3.x (typically bundled with Metashape).
- **Python Libraries**:
  - `opencv-python` (cv2)
  - `PyQt5` (for the GUI)
  - _(The script will attempt to install these automatically on first run if missing)_

## Installation

1.  Download the script file (`convert_to_cubemap_v012.py` or **recommended** `unified_fixed_v002.py`).
2.  On the first run, the script will check for necessary libraries (`opencv-python`, `PyQt5`).
3.  If libraries are missing, it will attempt to install them using `pip`. Administrator privileges or internet access might be required.
4.  If `pip` installation fails (e.g., due to network or permission issues), you may need to install the libraries manually into the Python environment used by Metashape.

## Usage

1.  Open your project in Agisoft Metashape Pro.
2.  Ensure you have an active chunk containing the spherical cameras you wish to convert.
3.  Run the script via the Metashape menu: `Tools -> Scripts -> Run Script...` and select the script file.

### For unified_fixed_v002.py (Recommended):

The fixed script runs directly with simple dialog prompts and:

- Analyzes spherical cameras automatically
- Converts to cubemap faces with correct geometry
- Creates properly oriented cameras in Metashape
- Exports COLMAP structure for 3D Gaussian Splatting
- **No realignment needed** - ready for 3DGS training

### Graphical User Interface (GUI) - v012:

If `PyQt5` is available, the graphical interface will launch:

- **Settings**:
  - `Output folder`: Specify the directory to save the generated face images.
  - `Overlap (degrees)`: Set the overlap angle between faces (useful for stitching later).
  - `Cube face size`: Select the resolution for each cube face. "Automatically" is recommended (uses source width / 4).
  - `Coordinate system`: Choose "Auto-detect" or manually select the project's coordinate system.
  - `Face processing threads`: Number of threads for parallel conversion of faces for **one** spherical camera.
  - `Camera processing threads`: Number of threads for parallel conversion of **different** spherical cameras simultaneously. **See the Threading Guide below!**
- **Image Parameters**:
  - `File format`: JPG, PNG, or TIFF.
  - `Quality`: For JPG (75-100).
  - `Interpolation`: Resampling method during remapping (Cubic is best quality, Nearest is fastest).
- **Cube Face Selection**: Check the boxes for the faces you want to generate.
- **Post-conversion Processing**:
  - `Realign cameras`: Run `Align Cameras` after adding the new cube cameras.
  - `Remove original spherical cameras`: Delete the original cameras after successful conversion.
- **Project Information**: Displays the number of cameras found and the detected coordinate system.
- **Processing Progress**: Shows overall progress, current operation, and estimated time remaining.
- **Control Buttons**: "Start", "Stop", "Close".

### Console Mode

If `PyQt5` is unavailable, the script will run in console mode, prompting for parameters sequentially via standard Metashape dialogs. Progress will be displayed in the console.

### Threading Guide

The script uses two parameters to control multithreading:

1.  **Face processing threads**:

    - **What it does**: Determines how many faces (e.g., front, back, top...) of a **single** spherical camera are processed in parallel during the image conversion stage (`cv2.remap`).
    - **Impact**: Increasing this speeds up the conversion of _each individual_ spherical camera but increases RAM and CPU load _during_ that camera's processing.
    - **Recommendations**: The default (`min(6, CPU_cores // 2)`) is a reasonable starting point. If you experience memory issues _during_ a single camera's conversion (before it's added to Metashape), try reducing this value (e.g., to 2-4).

2.  **Camera processing threads**:
    - **What it does**: Determines how many **different** spherical cameras are converted _simultaneously_. E.g., a value of 4 means the script attempts to convert four different spherical cameras in parallel.
    - **Impact**: Increasing this can significantly speed up the overall processing of _many_ cameras but **greatly increases peak RAM usage** as resources are needed for multiple simultaneous conversions.
    - **Recommendations**:
      - **For systems with ample RAM (e.g., 64GB+)**: You can experiment with values up to the number of CPU cores or slightly more if disk I/O is fast.
      - **❗️ For systems with limited RAM (e.g., 16-32GB or less) ❗️**: It is **strongly recommended to set this value to 1**. This forces the script to process spherical cameras **sequentially**, one after another. Peak RAM usage will then be primarily determined by processing the faces of _one_ camera (controlled by the first setting), which is much more stable.
      - **Successful low-RAM example**:
        - Camera processing threads: **1**
        - Face processing threads: **5** (or another moderate value like 4 or 6)
        - In this case, cameras are converted one by one, but each camera uses 5 threads to process its faces quickly.

## Localization

The script automatically detects the Metashape interface language (`ru` or `en`) and applies corresponding translations. English is used as a fallback for other languages.

## Troubleshooting

- **Dependency Installation Issues**: Ensure internet access and permissions to install Python packages. Try installing `opencv-python` and `PyQt5` manually via `pip` in Metashape's Python environment.
- **High RAM Usage / Crashes**: Reduce the number of threads, especially **"Camera processing threads"** (set it to **1**). You might also reduce "Face processing threads".
- **Path Issues (Cyrillic/Non-ASCII)**: The script includes path normalization, but if issues persist, ensure project and image paths do not contain highly unusual or invalid characters.
- **Geometric Issues**: If using v012 and experiencing incorrect camera orientations or projections, switch to `unified_fixed_v002.py` which has corrected mathematics.

## Version History

- **unified_fixed_v002 (Recommended)**:
  - **🔧 CRITICAL FIXES**: Corrected mathematical projection errors
  - **✅ Proper camera positioning**: All cubemap cameras positioned correctly
  - **✅ Fixed face orientations**: All 6 cube faces now show correct directions
  - **✅ Improved color extraction**: Better sparse point cloud color handling
  - **✅ COLMAP auto-export**: Ready structure for 3D Gaussian Splatting
  - **✅ No realignment needed**: Workflow simplified
- **v012 (Legacy)**:
  - Restored threaded camera processing (`ProcessCamerasThread`) for stability.
  - Added explicit memory management (`del`, `gc.collect()`) to reduce RAM usage.
  - Introduced separate controls for "Camera processing threads" and "Face processing threads".
  - Updated recommendations for thread settings, especially for low-RAM systems.
  - **⚠️ Known issues**: Mathematical projection errors, incorrect camera positioning
- **v0.11.x (Deprecated)**: Various changes including GUI improvements, face selection, post-processing options, Cyrillic path support, dependency installation, non-threaded experiments.
- **(Older versions)**: Basic console functionality.

## Acknowledgments

- Agisoft for the Metashape Python API.
- OpenCV library for image processing capabilities.
- PyQt5 framework for the graphical user interface.
