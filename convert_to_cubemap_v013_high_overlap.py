# === Spherical to Cubemap Converter (v0.1.2) ===
# Converts spherical images to cubemap format (6 faces)
# Optimized version with multithreading and GUI

import Metashape
import math
import sys
import os
import time
import traceback
import locale
import numpy as np
import cv2
import concurrent.futures
import gc

# === UTILITY FUNCTIONS ===

def setup_locale_and_encoding():
    """
    Sets up locale and encoding for correct handling of non-ASCII characters.
    """
    try:
        # Set locale to system default
        locale.setlocale(locale.LC_ALL, '')
        
        # Force UTF-8 for stdin/stdout/stderr if possible
        if sys.platform.startswith('win'):
            # Windows-specific console encoding fix
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
            except:
                pass
                
        # Reconfigure streams to UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
            
    except Exception as e:
        print(f"Warning: Failed to setup locale/encoding: {e}")

def fix_metashape_file_paths():
    """
    Monkey-patch Metashape methods to handle non-ASCII paths correctly if needed.
    """
    # In modern Metashape versions this might be less critical, but kept for compatibility
    pass

def normalize_path(path):
    """
    Normalizes file path: converts separators and handles absolute paths.
    """
    if path is None:
        return None
    return os.path.normpath(path)

def read_image_safe(path):
    """
    Reads image from path, handling non-ASCII characters correctly.
    """
    try:
        # For Windows with non-ASCII paths, use numpy fromfile
        if os.name == 'nt':
            with open(path, 'rb') as f:
                img_bytes = bytearray(f.read())
            nparr = np.asarray(img_bytes, dtype=np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return cv2.imread(path)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def save_image_safe(image, path, params=None):
    """
    Saves image to path, handling non-ASCII characters correctly.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.name == 'nt':
            ext = os.path.splitext(path)[1]
            result, img_bytes = cv2.imencode(ext, image, params if params else [])
            if result:
                with open(path, 'wb') as f:
                    f.write(img_bytes)
                return True
            return False
        else:
            return cv2.imwrite(path, image, params if params else [])
    except Exception as e:
        print(f"Error saving image {path}: {e}")
        return False

def get_string_option(label, options):
    """
    Helper to show a dropdown dialog in Metashape.
    """
    # Simple simulation of dropdown using input string if custom dialog not available
    # In a real script we might want a custom dialog, but here we use Metashape's simple input
    
    prompt = f"{label}\n"
    for i, opt in enumerate(options):
        prompt += f"{i+1}. {opt}\n"
    
    res = Metashape.app.getString(prompt, "1")
    try:
        idx = int(res) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except:
        pass
    return options[0] # Default to first

def console_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print new line on complete
    if iteration == total: 
        print()

# === PyQt5 IMPORTS ===
use_gui = True
try:
    if 'PyQt5' not in sys.modules:
        try:
            import PyQt5
        except ImportError:
            pass

    if 'PyQt5' in sys.modules:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                   QLabel, QPushButton, QProgressBar, QFileDialog, QCheckBox, 
                                   QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox)
        from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
        print("PyQt5 components imported successfully.")
    else:
        print("PyQt5 not found. GUI will be disabled.")
        use_gui = False
except ImportError as e:
    print(f"Failed to import PyQt5 components: {e}")
    use_gui = False
    print("Script will continue in console mode.")

print("Initialization complete.")

# === PROJECTION FUNCTIONS ===

def eqruirect2persp_map(img_shape, FOV, THETA, PHI, Hd, Wd, overlap=10):
    """
    Creates mapping maps for converting equirectangular projection to perspective.
    """
    equ_h, equ_w = img_shape
    equ_cx = (equ_w) / 2.0
    equ_cy = (equ_h) / 2.0

    wFOV = FOV + overlap
    hFOV = float(Hd) / Wd * wFOV

    c_x = (Wd) / 2.0
    c_y = (Hd) / 2.0

    w_len = 2 * np.tan(np.radians(wFOV / 2.0))
    w_interval = w_len / (Wd)

    h_len = 2 * np.tan(np.radians(hFOV / 2.0))
    h_interval = h_len / (Hd)

    x_map = np.zeros([Hd, Wd], np.float32) + 1
    y_map = np.tile((np.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
    z_map = -np.tile((np.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
    D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)

    xyz = np.zeros([Hd, Wd, 3], np.float64)
    xyz[:, :, 0] = (x_map / D)[:, :]
    xyz[:, :, 1] = (y_map / D)[:, :]
    xyz[:, :, 2] = (z_map / D)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, jacobian] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, jacobian] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([Hd * Wd, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2] / 1)
    lon = np.zeros([Hd * Wd], np.float64)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

    lon[idx1] = theta[idx1]
    lon[idx3] = theta[idx3] + np.pi
    lon[idx4] = theta[idx4] - np.pi

    lon = lon.reshape([Hd, Wd]) / np.pi * 180
    lat = -lat.reshape([Hd, Wd]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    return lon.astype(np.float32), lat.astype(np.float32)

def fix_back_face_artifact(image):
    """
    Fixes the black vertical strip artifact in the center of the image.
    """
    height, width = image.shape[:2]
    center_x = width // 2
    
    # Strip width to process
    strip_width = 3
    
    # Create copy for processing
    fixed_image = image.copy()
    
    # Define pixel range
    left_x = max(0, center_x - strip_width)
    right_x = min(width - 1, center_x + strip_width)
    
    # For each row
    for y in range(height):
        # Check for black pixels in central strip
        strip = image[y, left_x:right_x+1]
        
        # If too dark pixels found (possible artifact)
        if np.any(np.mean(strip, axis=1) < 30):
            # Use values from left and right of the strip for interpolation
            left_value = image[y, left_x-1] if left_x > 0 else image[y, right_x+1]
            right_value = image[y, right_x+1] if right_x < width-1 else image[y, left_x-1]
            
            # Linear interpolation
            for i, x in enumerate(range(left_x, right_x+1)):
                alpha = i / (right_x - left_x + 1)
                fixed_image[y, x] = (1 - alpha) * left_value + alpha * right_value
    
    # Apply smoothing only to fixed strip
    temp_mask = np.zeros_like(image)
    temp_mask[:, left_x:right_x+1] = 255
    
    # Gaussian blur on fixed region
    blur_region = cv2.GaussianBlur(fixed_image, (5, 5), 0)
    
    # Blend using mask
    mask = temp_mask.astype(np.float32) / 255.0
    blended = (mask * blur_region + (1.0 - mask) * fixed_image).astype(np.uint8)
    
    return blended
# === COORDINATE SYSTEM FUNCTIONS ===

def determine_coordinate_system():
    """
    Determines coordinate system type based on camera analysis.
    """
    print("Determining coordinate system type...")
    doc = Metashape.app.document
    chunk = doc.chunk
    cameras = [cam for cam in chunk.cameras if cam.transform]

    if not cameras:
        return "Y_UP"  # Default

    orientation_votes = {"Y_UP": 0, "Z_UP": 0, "X_UP": 0}

    for camera in cameras[:5]:
        rotation = camera.transform.rotation()
        up_vector = rotation * Metashape.Vector([0, 1, 0])

        y_alignment = abs(up_vector.y)
        z_alignment = abs(up_vector.z)
        x_alignment = abs(up_vector.x)

        if y_alignment > z_alignment and y_alignment > x_alignment:
            orientation_votes["Y_UP"] += 1
        elif z_alignment > y_alignment and z_alignment > x_alignment:
            orientation_votes["Z_UP"] += 1
        elif x_alignment > y_alignment and x_alignment > z_alignment:
            orientation_votes["X_UP"] += 1

    determined_orientation = max(orientation_votes, key=orientation_votes.get)
    print(f"Coordinate system determined: {determined_orientation}")
    return determined_orientation

# === CONVERSION LOGIC ===

def realign_cameras():
    """
    Re-aligns ALL cameras in the active chunk, resetting their current alignment.
    Uses parameters found by workers for API.
    Returns True if alignment was successfully started, False otherwise.
    """
    print("Re-aligning cameras...")
    try:
        # Check active chunk
        print("DEBUG: Checking active chunk...")
        chunk = Metashape.app.document.chunk
        if not chunk:
            raise Exception("Chunk not found. Please create or select a chunk.")
        print("DEBUG: Active chunk found.")

        # Try calling API with working parameters
        try:
            print("DEBUG: Calling chunk.matchPhotos() for all cameras...")
            chunk.matchPhotos(
                generic_preselection=True, 
                reference_preselection=False, 
                filter_mask=False, 
                keypoint_limit=100000, 
                tiepoint_limit=4000,
                guided_matching=False,
                reset_matches=True
            )
            print("DEBUG: chunk.matchPhotos() completed.")
            
            print("DEBUG: Calling chunk.alignCameras() for all cameras...")
            chunk.alignCameras(
                adaptive_fitting=False, 
                reset_alignment=True
            )
            print("DEBUG: chunk.alignCameras() completed.")
            print("Alignment completed successfully.")
            return True
        
        except Exception as api_error:
            error_message = f"Re-alignment error: {str(api_error)}"
            print(f"DEBUG: API Error (matchPhotos/alignCameras): {api_error}")
            print(error_message)
            print(traceback.format_exc())
            if Metashape.app is not None:
                Metashape.app.messageBox(f"Error\n\n{error_message}\n\n{traceback.format_exc()}")
            return False

    except Exception as e:
        error_message = f"Re-alignment error: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        if Metashape.app is not None:
            Metashape.app.messageBox(f"Error\n\n{error_message}\n\n{traceback.format_exc()}")
        return False

def remove_spherical_cameras():
    """
    Removes source spherical cameras from the project.
    """
    print("Removing spherical cameras...")
    try:
        doc = Metashape.app.document
        chunk = doc.chunk
        
        # Get list of spherical cameras (those without face suffixes)
        spherical_cameras = []
        for camera in chunk.cameras:
            is_spherical = True
            for suffix in ["_front", "_right", "_left", "_top", "_down", "_back"]:
                if camera.label.endswith(suffix):
                    is_spherical = False
                    break
            if is_spherical:
                spherical_cameras.append(camera)
        
        # Remove found cameras
        for camera in spherical_cameras:
            chunk.remove(camera)
        
        print("Removal completed.")
        return True
    except Exception as e:
        print(f"Error removing spherical cameras: {str(e)}")
        return False

def convert_spherical_to_cubemap(spherical_image_path, output_folder, camera_label, persp_size=None, overlap=10, 
                                file_format="jpg", quality=95, interpolation=cv2.INTER_CUBIC, max_workers=None,
                                selected_faces=None):
    """
    Converts spherical image to cubemap projection.
    Uses multithreading for faster processing.
    """
    print("Converting spherical image to cubemap projection...")
    
    if max_workers is None:
        max_workers = min(6, os.cpu_count() or 1)
    
    print(f"Using {max_workers} threads for parallel face processing...")
        
    normalized_image_path = normalize_path(spherical_image_path)
    normalized_output_folder = normalize_path(output_folder)
    
    try:
        os.makedirs(normalized_output_folder, exist_ok=True)
        print(f"Creating directory: {normalized_output_folder}")
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
    
    spherical_image = read_image_safe(normalized_image_path)
    if spherical_image is None:
        raise ValueError(f"Failed to load image: {spherical_image_path}")

    equirect_height, equirect_width = spherical_image.shape[:2]
    
    if persp_size is None:
        # Use approx 1/4 of source width
        persp_size = min(max(equirect_width // 4, 512), 4096)
        print(f"Automatically calculated face size: {persp_size}px")

    all_faces = ["front", "right", "left", "top", "down", "back"]
    faces_to_process = selected_faces if selected_faces else all_faces
    
    if selected_faces:
        print(f"Selected faces: {', '.join(faces_to_process)}")
    else:
        print("No faces selected (or all faces).")
    
    faces_params = {
        "front": {"fov": 90, "theta": 0, "phi": 0},
        "right": {"fov": 90, "theta": 90, "phi": 0},
        "left": {"fov": 90, "theta": -90, "phi": 0},
        "top": {"fov": 90, "theta": 0, "phi": 90},
        "down": {"fov": 90, "theta": 0, "phi": -90},
        "back": {"fov": 90, "theta": 180, "phi": 0},
        # High Overlap (10 faces) additions
        "front_right": {"fov": 90, "theta": 45, "phi": 0},
        "back_right": {"fov": 90, "theta": 135, "phi": 0},
        "back_left": {"fov": 90, "theta": -135, "phi": 0},
        "front_left": {"fov": 90, "theta": -45, "phi": 0},
    }
    
    faces_params = {face: params for face, params in faces_params.items() if face in faces_to_process}

    image_paths = {}
    
    save_params = []
    file_ext = file_format.lower()
    
    if file_ext == "jpg" or file_ext == "jpeg":
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        file_ext = "jpg"
    elif file_ext == "png":
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, 10 - quality//10)]
        file_ext = "png"
    elif file_ext == "tiff" or file_ext == "tif":
        save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        file_ext = "tiff"
    else:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        file_ext = "jpg"

    def process_face(face_name, params):
        try:
            print(f"Processing face {face_name}...")
            
            map_x, map_y = eqruirect2persp_map(
                img_shape=(equirect_height, equirect_width),
                FOV=params["fov"],
                THETA=params["theta"],
                PHI=params["phi"],
                Hd=persp_size,
                Wd=persp_size,
                overlap=overlap
            )
    
            perspective_image = cv2.remap(spherical_image, map_x, map_y, interpolation=interpolation)
        
            if face_name == "back":
                perspective_image = fix_back_face_artifact(perspective_image)
            
            output_filename = f"{camera_label}_{face_name}.{file_ext}"
            output_path = os.path.join(normalized_output_folder, output_filename)
            
            success = save_image_safe(perspective_image, output_path, save_params)
            
            if not success:
                raise ValueError(f"Failed to save image {output_path}")
            
            print(f"Face {face_name} successfully processed")

            del map_x, map_y, perspective_image

            return face_name, output_path
        except Exception as e:
            print(f"Error processing face {face_name}: {str(e)}")
            if 'map_x' in locals(): del map_x
            if 'map_y' in locals(): del map_y
            if 'perspective_image' in locals(): del perspective_image
            return face_name, None
    
    if not faces_params:
        return image_paths
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_face = {executor.submit(process_face, face_name, params): face_name 
                          for face_name, params in faces_params.items()}
        
        for future in concurrent.futures.as_completed(future_to_face):
            face_name = future_to_face[future]
            try:
                face_name, path = future.result()
                if path:
                    image_paths[face_name] = path
            except Exception as e:
                print(f"Error processing face {face_name}: {str(e)}")
    
    if not image_paths and selected_faces:
        raise ValueError("Failed to create any cube faces.")
        
    return image_paths
# === CAMERA CREATION FUNCTIONS ===

def add_cubemap_cameras(chunk, spherical_camera, image_paths, persp_size, coord_system="Y_UP"):
    """
    Adds cameras for cubemap faces based on spherical camera position.
    """
    print("Adding cameras based on spherical camera...")
    
    # Source spherical camera position
    position = spherical_camera.transform.translation()

    # Source spherical camera orientation
    base_rotation = spherical_camera.transform.rotation()

    def convert_to_4x4(matrix_3x3):
        return Metashape.Matrix([
            [matrix_3x3[0, 0], matrix_3x3[0, 1], matrix_3x3[0, 2], 0],
            [matrix_3x3[1, 0], matrix_3x3[1, 1], matrix_3x3[1, 2], 0],
            [matrix_3x3[2, 0], matrix_3x3[2, 1], matrix_3x3[2, 2], 0],
            [0, 0, 0, 1]
        ])

    base_rotation_4x4 = convert_to_4x4(base_rotation)

    cubemap_directions = {}
    
    if coord_system == "Y_UP":
        print("Setting up directions for cube faces (Y_UP)...")
        cubemap_directions = {
            "front": {"forward": [0, 0, 1], "up": [0, 1, 0]},
            "right": {"forward": [1, 0, 0], "up": [0, 1, 0]},
            "left": {"forward": [-1, 0, 0], "up": [0, 1, 0]},
            "top": {"forward": [0, 1, 0], "up": [0, 0, -1]},
            "down": {"forward": [0, -1, 0], "up": [0, 0, 1]},
            "back": {"forward": [0, 0, -1], "up": [0, 1, 0]},
            # High Overlap
            "front_right": {"forward": [1, 0, 1], "up": [0, 1, 0]},
            "back_right": {"forward": [1, 0, -1], "up": [0, 1, 0]},
            "back_left": {"forward": [-1, 0, -1], "up": [0, 1, 0]},
            "front_left": {"forward": [-1, 0, 1], "up": [0, 1, 0]},
        }
    elif coord_system == "Z_UP":
        print("Setting up directions for cube faces (Z_UP)...")
        cubemap_directions = {
            "front": {"forward": [1, 0, 0], "up": [0, 0, 1]},
            "right": {"forward": [0, 1, 0], "up": [0, 0, 1]},
            "left": {"forward": [0, -1, 0], "up": [0, 0, 1]},
            "top": {"forward": [0, 0, 1], "up": [-1, 0, 0]},
            "down": {"forward": [0, 0, -1], "up": [1, 0, 0]},
            "back": {"forward": [-1, 0, 0], "up": [0, 0, 1]},
            # High Overlap
            "front_right": {"forward": [1, 1, 0], "up": [0, 0, 1]},
            "back_right": {"forward": [-1, 1, 0], "up": [0, 0, 1]},
            "back_left": {"forward": [-1, -1, 0], "up": [0, 0, 1]},
            "front_left": {"forward": [1, -1, 0], "up": [0, 0, 1]},
        }
    elif coord_system == "X_UP":
        print("Setting up directions for cube faces (X_UP)...")
        cubemap_directions = {
            "front": {"forward": [0, 1, 0], "up": [1, 0, 0]},
            "right": {"forward": [0, 0, 1], "up": [1, 0, 0]},
            "left": {"forward": [0, 0, -1], "up": [1, 0, 0]},
            "top": {"forward": [1, 0, 0], "up": [0, -1, 0]},
            "down": {"forward": [-1, 0, 0], "up": [0, 1, 0]},
            "back": {"forward": [0, -1, 0], "up": [1, 0, 0]},
            # High Overlap
            "front_right": {"forward": [0, 1, 1], "up": [1, 0, 0]},
            "back_right": {"forward": [0, -1, 1], "up": [1, 0, 0]},
            "back_left": {"forward": [0, -1, -1], "up": [1, 0, 0]},
            "front_left": {"forward": [0, 1, -1], "up": [1, 0, 0]},
        }
    else:
        print(f"Warning: unknown coordinate system '{coord_system}'. Using Y_UP.")
        cubemap_directions = {
            "front": {"forward": [0, 0, 1], "up": [0, 1, 0]},
            "right": {"forward": [1, 0, 0], "up": [0, 1, 0]},
            "left": {"forward": [-1, 0, 0], "up": [0, 1, 0]},
            "top": {"forward": [0, 1, 0], "up": [0, 0, -1]},
            "down": {"forward": [0, -1, 0], "up": [0, 0, 1]},
            "back": {"forward": [0, 0, -1], "up": [0, 1, 0]},
            # High Overlap
            "front_right": {"forward": [1, 0, 1], "up": [0, 1, 0]},
            "back_right": {"forward": [1, 0, -1], "up": [0, 1, 0]},
            "back_left": {"forward": [-1, 0, -1], "up": [0, 1, 0]},
            "front_left": {"forward": [-1, 0, 1], "up": [0, 1, 0]},
        }

    def create_rotation_matrix(forward, up, base_rotation_4x4):
        forward = Metashape.Vector(forward).normalized()
        up = Metashape.Vector(up)
        right = Metashape.Vector.cross(forward, up).normalized()
        up = Metashape.Vector.cross(right, forward).normalized()

        rotation_matrix = Metashape.Matrix([
            [right.x, right.y, right.z, 0],
            [up.x, up.y, up.z, 0],
            [forward.x, forward.y, forward.z, 0],
            [0, 0, 0, 1]
        ])

        return base_rotation_4x4 * rotation_matrix

    cameras_created = []
    print(f"DEBUG: Adding cameras for faces: {list(image_paths.keys())}")
    for face_name in image_paths.keys():
        if face_name not in cubemap_directions:
            print(f"DEBUG: Skipping face {face_name}, no directions found.")
            continue

        print(f"DEBUG: Processing face: {face_name}")
        directions = cubemap_directions[face_name]

        print(f"DEBUG: Calling chunk.addCamera() for {face_name}...")
        camera = None
        try:
            camera = chunk.addCamera()
            if camera is None:
                print(f"ERROR: chunk.addCamera() returned None for face {face_name}")
                continue
        except Exception as e:
            print(f"ERROR: Exception during chunk.addCamera() for face {face_name}: {e}")
            traceback.print_exc()
            continue

        camera.label = f"{spherical_camera.label}_{face_name}"
        print(f"Added camera {camera.label}")

        print(f"DEBUG: Finding/creating sensor for {face_name}...")
        persp_sensors = [s for s in chunk.sensors if s.type == Metashape.Sensor.Type.Frame and s.width == persp_size and s.height == persp_size]
        if persp_sensors:
            camera.sensor = persp_sensors[0]
            print(f"DEBUG: Reusing existing sensor '{persp_sensors[0].label}'")
        else:
            print(f"DEBUG: Creating new sensor for {face_name}...")
            sensor = chunk.addSensor()
            if sensor is None:
                print(f"ERROR: chunk.addSensor() returned None for face {face_name}")
                continue
            sensor.label = f"Perspective_{persp_size}px"
            sensor.type = Metashape.Sensor.Type.Frame
            sensor.width = persp_size
            sensor.height = persp_size
            camera.sensor = sensor
            print(f"Created sensor {sensor.label}")

        cameras_created.append(camera)

        print(f"DEBUG: Setting sensor parameters for {face_name}...")
        sensor = camera.sensor
        sensor.type = Metashape.Sensor.Type.Frame
        sensor.width = persp_size
        sensor.height = persp_size

        focal_length = persp_size / (2 * np.tan(np.radians(90 / 2)))
        sensor.focal_length = focal_length
        sensor.pixel_width = 1
        sensor.pixel_height = 1

        print(f"DEBUG: Setting calibration for {face_name}...")
        calibration = sensor.calibration
        calibration.f = focal_length
        calibration.cx = persp_size / 2
        calibration.cy = persp_size / 2
        calibration.k1 = 0
        calibration.k2 = 0
        calibration.k3 = 0
        calibration.p1 = 0
        calibration.p2 = 0

        print(f"DEBUG: Setting transform (translation) for {face_name}...")
        camera.transform = Metashape.Matrix.Translation(position)

        print(f"DEBUG: Calculating rotation matrix for {face_name}...")
        forward = directions["forward"]
        up = directions["up"]
        rotation_matrix = create_rotation_matrix(forward, up, base_rotation_4x4)
        print(f"Created rotation matrix for face {face_name}")
        print(f"DEBUG: Setting transform (rotation) for {face_name}...")
        camera.transform = camera.transform * rotation_matrix

        print(f"DEBUG: Creating Photo object for {face_name}...")
        camera.photo = Metashape.Photo()

        normalized_path = normalize_path(image_paths[face_name])
        print(f"DEBUG: Setting photo path '{normalized_path}' for {face_name}...")
        camera.photo.path = normalized_path

        print(f"DEBUG: Checking if file exists for {face_name}...")
        if not os.path.exists(normalized_path):
            print(f"Error: image file not found: {normalized_path}")
            continue

        print(f"DEBUG: Setting metadata for {face_name}...")
        camera.meta['Image/Width'] = str(persp_size)
        camera.meta['Image/Height'] = str(persp_size)
        camera.meta['Image/Orientation'] = "1"
        print(f"Set metadata for camera {camera.label}")
        print(f"DEBUG: Finished processing face {face_name}")

    print("DEBUG: Finished add_cubemap_cameras function.")
    return cameras_created

# === THREADING CLASSES ===

if 'PyQt5' in sys.modules:
    from PyQt5.QtCore import QThread, pyqtSignal
    
    class ProcessCamerasThread(QThread):
        update_progress = pyqtSignal(int, int, str, str, int)  # progress, total, camera_name, status, percent
        processing_finished = pyqtSignal(bool, dict)  # success, stats
        error_occurred = pyqtSignal(str)  # error message
        
        def __init__(self, cameras, output_folder, options):
            super().__init__()
            self.cameras = cameras
            self.output_folder = output_folder
            self.options = options
            self.stop_requested = False
            self.faces_threads = self.options.get("faces_threads", min(6, os.cpu_count() or 1))
            default_camera_threads = max(1, (os.cpu_count() // 2) if os.cpu_count() else 1)
            self.camera_threads = self.options.get("camera_threads", default_camera_threads) 
        
        def _process_single_camera(self, camera):
            """Executes conversion for a single camera."""
            if self.stop_requested:
                return None

            camera_label = camera.label
            spherical_image_path = camera.photo.path
            normalized_path = normalize_path(spherical_image_path)

            try:
                selected_faces = self.options.get("selected_faces", None)
                image_paths = convert_spherical_to_cubemap(
                    spherical_image_path=normalized_path,
                    output_folder=self.output_folder,
                    camera_label=camera_label,
                    persp_size=self.options.get("persp_size"),
                    overlap=self.options.get("overlap", 10),
                    file_format=self.options.get("file_format", "jpg"),
                    quality=self.options.get("quality", 95),
                    interpolation=self.options.get("interpolation", cv2.INTER_CUBIC),
                    max_workers=self.faces_threads,
                    selected_faces=selected_faces
                )

                if image_paths:
                    first_image = list(image_paths.values())[0]
                    image = read_image_safe(first_image)
                    if image is None:
                        print(f"Warning: Failed to read image {first_image} for camera {camera_label} after conversion.")
                        actual_size = None
                    else:
                        actual_size = image.shape[0]

                    return {
                        "camera": camera,
                        "image_paths": image_paths,
                        "actual_size": actual_size,
                        "error": None
                    }
                else:
                    print(f"No faces created for camera {camera_label}")
                    return {
                        "camera": camera,
                        "image_paths": None,
                        "actual_size": None,
                        "error": "skipped_no_faces"
                    }

            except Exception as e:
                error_message = f"Conversion error: {camera_label}: {str(e)}"
                print(error_message)
                print(traceback.format_exc())
                return {
                    "camera": camera,
                    "image_paths": None,
                    "actual_size": None,
                    "error": error_message
                }

        def run(self):
            try:
                print("Starting GUI processing thread...")
                start_time = time.time()
                total_cameras = len(self.cameras)
                processed_count = 0
                skipped_count = 0
                conversion_errors = []
                add_camera_errors = []

                print(f"--- Stage 1: Image Conversion ({self.camera_threads} threads) ---")
                conversion_results = []
                futures = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.camera_threads) as executor:
                    self.update_progress.emit(0, total_cameras, "", "Submitting conversion tasks...", 0)

                    for i, camera in enumerate(self.cameras):
                        if self.stop_requested:
                            print("Stop requested before submitting all conversion tasks.")
                            break
                        future = executor.submit(self._process_single_camera, camera)
                        futures.append(future)

                    self.update_progress.emit(0, total_cameras, "", "Collecting conversion results...", 5)
                    conversion_completed_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        if self.stop_requested:
                            print("Stop requested during result collection.")
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break

                        result = future.result()
                        conversion_completed_count += 1
                        progress_percent_conv = 5 + int((conversion_completed_count / total_cameras) * 45)

                        if result:
                             camera_label = result["camera"].label
                             if result["error"]:
                                 skipped_count += 1
                                 if result["error"] != "skipped_no_faces":
                                     conversion_errors.append(result["error"])
                                 status_msg = f"Error: {camera_label}" if result["error"] != "skipped_no_faces" else f"Skipped (no faces): {camera_label}"
                                 self.update_progress.emit(
                                     conversion_completed_count, total_cameras, camera_label,
                                     status_msg,
                                     progress_percent_conv
                                 )
                             elif result["actual_size"] is None:
                                 skipped_count += 1
                                 error_msg = f"Error reading image after conversion for {camera_label}"
                                 conversion_errors.append(error_msg)
                                 self.update_progress.emit(
                                     conversion_completed_count, total_cameras, camera_label,
                                     f"Error: {camera_label}",
                                     progress_percent_conv
                                 )
                             else:
                                 conversion_results.append(result)
                                 self.update_progress.emit(
                                     conversion_completed_count, total_cameras, camera_label,
                                     f"Converted: {camera_label}",
                                     progress_percent_conv
                                 )
                        else:
                            print(f"Camera processing interrupted (result None).")

                if self.stop_requested:
                    print("Processing aborted (during conversion)")
                    self.processing_finished.emit(False, {
                        "processed": processed_count, "skipped": skipped_count,
                        "total": total_cameras, "time": time.time() - start_time,
                        "errors": conversion_errors + add_camera_errors
                    })
                    return

                print("--- Stage 2: Adding cameras to Metashape (Sequential) ---")
                total_to_add = len(conversion_results)
                added_count = 0
                if total_to_add > 0:
                    chunk = Metashape.app.document.chunk
                    coord_system = self.options.get("coord_system", "Y_UP")

                    for result in conversion_results:
                        if self.stop_requested:
                            print("Processing aborted (during camera addition)")
                            break

                        camera = result["camera"]
                        camera_label = camera.label
                        image_paths = result["image_paths"]
                        actual_size = result["actual_size"]

                        progress_percent_add = 50 + int((added_count / total_to_add) * 40)
                        self.update_progress.emit(
                            conversion_completed_count + added_count,
                            total_cameras,
                            camera_label,
                            f"Adding cameras for {camera_label} ({added_count + 1}/{total_to_add})...",
                            progress_percent_add
                        )

                        try:
                            add_cubemap_cameras(
                                chunk=chunk,
                                spherical_camera=camera,
                                image_paths=image_paths,
                                persp_size=actual_size,
                                coord_system=coord_system
                            )
                            processed_count += 1
                            self.update_progress.emit(
                                conversion_completed_count + added_count + 1, total_cameras,
                                camera_label,
                                f"Added: {camera_label}",
                                progress_percent_add
                            )
                            print(f"Camera {camera_label} successfully processed")

                        except Exception as e:
                            error_message = f"Error adding camera {camera_label}: {str(e)}"
                            print(error_message)
                            print(traceback.format_exc())
                            add_camera_errors.append(error_message)
                            self.update_progress.emit(
                                conversion_completed_count + added_count + 1, total_cameras,
                                camera_label,
                                f"Error adding: {camera_label}",
                                progress_percent_add
                            )
                        finally:
                             added_count += 1
                             gc.collect()

                if self.stop_requested:
                     print("Processing aborted (during camera addition)")
                     self.processing_finished.emit(False, {
                         "processed": processed_count,
                         "skipped": skipped_count + (total_to_add - added_count),
                         "total": total_cameras,
                         "time": time.time() - start_time,
                         "errors": conversion_errors + add_camera_errors
                     })
                     return

                print("--- Stage 3: Post-processing (Sequential) ---")
                final_progress = 90

                if self.options.get("realign_cameras_after", False):
                    if self.stop_requested: return
                    try:
                        self.update_progress.emit(total_cameras, total_cameras, "", "Re-aligning cameras...", final_progress)
                        realign_cameras()
                        final_progress += 5
                    except Exception as e:
                        error_message = f"Re-alignment error: {str(e)}"
                        print(error_message); add_camera_errors.append(error_message)

                if self.options.get("remove_spherical_cameras_after", False):
                    if self.stop_requested: return
                    try:
                        self.update_progress.emit(total_cameras, total_cameras, "", "Removing spherical cameras...", final_progress)
                        remove_spherical_cameras()
                        final_progress += 5
                    except Exception as e:
                        error_message = f"Error removing spherical cameras: {str(e)}"
                        print(error_message); add_camera_errors.append(error_message)

                self.update_progress.emit(total_cameras, total_cameras, "", "Processing finished", 100)
                total_time = time.time() - start_time
                final_skipped_count = skipped_count + (total_to_add - processed_count)
                print(f"Processing stats: processed {processed_count} of {total_cameras}, skipped {final_skipped_count}")
                
                self.processing_finished.emit(True, {
                    "processed": processed_count,
                    "skipped": final_skipped_count,
                    "total": total_cameras,
                    "time": total_time,
                    "errors": conversion_errors + add_camera_errors
                })
                print("GUI processing thread finished")

            except Exception as e:
                error_message = f"General processing error: {str(e)}"
                print(error_message)
                print(traceback.format_exc())
                self.error_occurred.emit(error_message)
                print(f"Error in GUI thread: {str(e)}")

        def stop(self):
            self.stop_requested = True
            print("Processing aborted")
# === GUI CLASS ===

if 'PyQt5' in sys.modules:
    class CubemapConverterGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.cpu_count = os.cpu_count() or 1
            self.init_ui()
            self.process_thread = None

        def init_ui(self):
            """Initialize User Interface"""
            self.setWindowTitle("Spherical to Cubemap Image Conversion")
            self.setGeometry(100, 100, 800, 650)

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)

            # Settings Group
            settings_group = QGroupBox("Settings")
            settings_layout = QVBoxLayout()

            # Output Folder
            output_folder_layout = QHBoxLayout()
            self.output_folder_label = QLabel("Output folder:")
            self.output_folder_path = QLabel("Not selected")
            self.output_folder_path.setStyleSheet("font-weight: bold;")
            self.browse_button = QPushButton("Browse...")
            self.browse_button.clicked.connect(self.select_output_folder)

            output_folder_layout.addWidget(self.output_folder_label)
            output_folder_layout.addWidget(self.output_folder_path, 1)
            output_folder_layout.addWidget(self.browse_button)
            settings_layout.addLayout(output_folder_layout)

            # Overlap
            overlap_layout = QHBoxLayout()
            overlap_label = QLabel("Overlap (degrees):")
            self.overlap_spinner = QDoubleSpinBox()
            self.overlap_spinner.setRange(0, 20)
            self.overlap_spinner.setValue(10)
            self.overlap_spinner.setDecimals(1)
            self.overlap_spinner.setSingleStep(0.5)

            overlap_layout.addWidget(overlap_label)
            overlap_layout.addWidget(self.overlap_spinner)
            settings_layout.addLayout(overlap_layout)

            # Face Size
            size_layout = QHBoxLayout()
            size_label = QLabel("Cube face size:")
            self.size_combo = QComboBox()
            self.size_combo.addItem("Automatically (recommended)", None)
            for size in [512, 1024, 2048, 4096]:
                self.size_combo.addItem(f"{size}x{size}", size)
            self.size_combo.setCurrentIndex(0)

            size_layout.addWidget(size_label)
            size_layout.addWidget(self.size_combo)
            settings_layout.addLayout(size_layout)

            # Coordinate System
            coord_system_layout = QHBoxLayout()
            coord_system_label = QLabel("Coordinate system:")
            self.coord_system_combo = QComboBox()
            self.coord_system_combo.addItems(["Y_UP", "Z_UP", "X_UP", "Auto-detect"])
            self.coord_system_combo.setCurrentText("Auto-detect")

            coord_system_layout.addWidget(coord_system_label)
            coord_system_layout.addWidget(self.coord_system_combo)
            settings_layout.addLayout(coord_system_layout)
            
            # Threading Settings
            thread_layout = QHBoxLayout()
            thread_label = QLabel("Face processing threads:")
            self.thread_spinner = QSpinBox()
            self.thread_spinner.setRange(1, os.cpu_count() or 4)
            self.thread_spinner.setValue(min(6, os.cpu_count() or 1))
            self.thread_spinner.setToolTip("Number of parallel threads for processing faces of ONE camera")
            
            thread_layout.addWidget(thread_label)
            thread_layout.addWidget(self.thread_spinner)
            settings_layout.addLayout(thread_layout)
            
            camera_thread_layout = QHBoxLayout()
            camera_thread_label = QLabel("Camera processing threads:")
            self.camera_thread_spinner = QSpinBox()
            default_camera_threads = max(1, (self.cpu_count // 2)) 
            self.camera_thread_spinner.setRange(1, self.cpu_count) 
            self.camera_thread_spinner.setValue(default_camera_threads) 
            self.camera_thread_spinner.setToolTip("Number of parallel threads for converting DIFFERENT cameras")
            
            camera_thread_layout.addWidget(camera_thread_label)
            camera_thread_layout.addWidget(self.camera_thread_spinner)
            settings_layout.addLayout(camera_thread_layout)

            settings_group.setLayout(settings_layout)
            main_layout.addWidget(settings_group)
            
            # Image Parameters Group
            image_group = QGroupBox("Image Parameters")
            image_layout = QVBoxLayout()
            
            # File Format
            format_layout = QHBoxLayout()
            format_label = QLabel("File format:")
            self.format_combo = QComboBox()
            self.format_combo.addItems(["JPEG (JPG)", "PNG", "TIFF"])
            self.format_combo.setCurrentIndex(0)
            
            format_layout.addWidget(format_label)
            format_layout.addWidget(self.format_combo)
            image_layout.addLayout(format_layout)
            
            # Quality
            quality_layout = QHBoxLayout()
            quality_label = QLabel("Quality:")
            self.quality_spinner = QSpinBox()
            self.quality_spinner.setRange(75, 100)
            self.quality_spinner.setValue(95)
            self.quality_spinner.setSingleStep(1)
            
            quality_layout.addWidget(quality_label)
            quality_layout.addWidget(self.quality_spinner)
            image_layout.addLayout(quality_layout)
            
            # Interpolation
            interp_layout = QHBoxLayout()
            interp_label = QLabel("Interpolation:")
            self.interp_combo = QComboBox()
            self.interp_combo.addItem("Nearest (faster, lower quality)", cv2.INTER_NEAREST)
            self.interp_combo.addItem("Linear (average)", cv2.INTER_LINEAR)
            self.interp_combo.addItem("Cubic (slower, better quality)", cv2.INTER_CUBIC)
            self.interp_combo.setCurrentIndex(2)
            
            interp_layout.addWidget(interp_label)
            interp_layout.addWidget(self.interp_combo)
            image_layout.addLayout(interp_layout)
            
            image_group.setLayout(image_layout)
            main_layout.addWidget(image_group)
            
            # Layout Selection
            layout_group = QHBoxLayout()
            layout_label = QLabel("Layout:")
            self.layout_combo = QComboBox()
            self.layout_combo.addItem("Standard (6 faces)", "standard")
            self.layout_combo.addItem("High Overlap (10 faces)", "high_overlap")
            self.layout_combo.currentIndexChanged.connect(self.update_face_selection)
            
            layout_group.addWidget(layout_label)
            layout_group.addWidget(self.layout_combo)
            settings_layout.addLayout(layout_group)

            # Face Selection Group
            faces_group = QGroupBox("Select Faces")
            faces_layout = QVBoxLayout()
        
            faces_label = QLabel("Select faces to generate:")
            faces_layout.addWidget(faces_label)
        
            self.face_checkboxes = {}
            # Standard faces
            self.standard_faces = ["front", "right", "left", "top", "down", "back"]
            # Extra faces for high overlap
            self.extra_faces = ["front_right", "back_right", "back_left", "front_left"]
            
            # Create checkboxes for all potential faces
            all_faces_display = [
                ("front", "Front (0°)"),
                ("front_right", "Front-Right (45°)"),
                ("right", "Right (90°)"),
                ("back_right", "Back-Right (135°)"),
                ("back", "Back (180°)"),
                ("back_left", "Back-Left (-135°)"),
                ("left", "Left (-90°)"),
                ("front_left", "Front-Left (-45°)"),
                ("top", "Top (+90°)"),
                ("down", "Down (-90°)")
            ]
            
            for face_key, face_label in all_faces_display:
                checkbox = QCheckBox(face_label)
                checkbox.setChecked(True)
                self.face_checkboxes[face_key] = checkbox
                faces_layout.addWidget(checkbox)
                
            faces_group.setLayout(faces_layout)
            main_layout.addWidget(faces_group)
            
            # Initial update to hide extra faces
            self.update_face_selection()
        
            # Post-processing Group
            post_group = QGroupBox("Post-processing")
            post_layout = QVBoxLayout()
        
            self.realign_checkbox = QCheckBox("Re-align cameras after conversion")
            post_layout.addWidget(self.realign_checkbox)
        
            self.remove_spherical_checkbox = QCheckBox("Remove source spherical cameras")
            post_layout.addWidget(self.remove_spherical_checkbox)
        
            post_group.setLayout(post_layout)
            main_layout.addWidget(post_group)
            
            # Project Info Group
            info_group = QGroupBox("Project Information")
            info_layout = QVBoxLayout()
            
            camera_count_layout = QHBoxLayout()
            camera_count_label = QLabel("Number of detected cameras:")
            self.camera_count_value = QLabel("0")
            self.camera_count_value.setStyleSheet("font-weight: bold;")
            camera_count_layout.addWidget(camera_count_label)
            camera_count_layout.addWidget(self.camera_count_value)
            info_layout.addLayout(camera_count_layout)
            
            detected_system_layout = QHBoxLayout()
            detected_system_label = QLabel("Detected coordinate system:")
            self.detected_system_value = QLabel("Not detected")
            self.detected_system_value.setStyleSheet("font-weight: bold;")
            detected_system_layout.addWidget(detected_system_label)
            detected_system_layout.addWidget(self.detected_system_value)
            info_layout.addLayout(detected_system_layout)
            
            info_group.setLayout(info_layout)
            main_layout.addWidget(info_group)
            
            # Progress Group
            progress_group = QGroupBox("Processing Progress")
            progress_layout = QVBoxLayout()
            
            self.progress_bar = QProgressBar()
            progress_layout.addWidget(self.progress_bar)
            
            current_camera_layout = QHBoxLayout()
            current_camera_label = QLabel("Current camera:")
            self.current_camera_value = QLabel("None")
            current_camera_layout.addWidget(current_camera_label)
            current_camera_layout.addWidget(self.current_camera_value, 1)
            progress_layout.addLayout(current_camera_layout)
            
            status_layout = QHBoxLayout()
            status_label = QLabel("Status:")
            self.status_value = QLabel("Waiting to start")
            status_layout.addWidget(status_label)
            status_layout.addWidget(self.status_value, 1)
            progress_layout.addLayout(status_layout)
            
            time_layout = QHBoxLayout()
            time_label = QLabel("Time remaining:")
            self.time_value = QLabel("--:--:--")
            time_layout.addWidget(time_label)
            time_layout.addWidget(self.time_value)
            progress_layout.addLayout(time_layout)
            
            progress_group.setLayout(progress_layout)
            main_layout.addWidget(progress_group)
            
            # Buttons
            buttons_layout = QHBoxLayout()
            
            self.start_button = QPushButton("Start")
            self.start_button.clicked.connect(self.start_processing)
            self.start_button.setMinimumWidth(120)
            
            self.stop_button = QPushButton("Stop")
            self.stop_button.clicked.connect(self.stop_processing)
            self.stop_button.setEnabled(False)
            self.stop_button.setMinimumWidth(120)
            
            self.close_button = QPushButton("Close")
            self.close_button.clicked.connect(self.close)
            self.close_button.setMinimumWidth(120)
            
            buttons_layout.addStretch()
            buttons_layout.addWidget(self.start_button)
            buttons_layout.addWidget(self.stop_button)
            buttons_layout.addWidget(self.close_button)
            buttons_layout.addStretch()
            
            main_layout.addLayout(buttons_layout)
            
            self.status_bar = self.statusBar()
            self.status_bar.showMessage("Ready")
            
            QTimer.singleShot(100, self.update_project_info)
        
        def update_face_selection(self):
            layout = self.layout_combo.currentData()
            
            # Show/Hide checkboxes based on layout
            for key, checkbox in self.face_checkboxes.items():
                if layout == "standard":
                    if key in self.standard_faces:
                        checkbox.setVisible(True)
                        checkbox.setChecked(True)
                    else:
                        checkbox.setVisible(False)
                        checkbox.setChecked(False)
                elif layout == "high_overlap":
                    checkbox.setVisible(True)
                    checkbox.setChecked(True)

        def select_output_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Select output folder for images")
            if folder:
                self.output_folder_path.setText(normalize_path(folder))
                
        def update_project_info(self):
            try:
                doc = Metashape.app.document
                chunk = doc.chunk
                
                if not chunk:
                    QMessageBox.warning(self, "Error", "Active chunk not found.")
                    return
                
                spherical_cameras = [cam for cam in chunk.cameras if cam.transform and cam.photo]
                self.camera_count_value.setText(str(len(spherical_cameras)))
                
                if spherical_cameras:
                    try:
                        coord_system = determine_coordinate_system()
                        self.detected_system_value.setText(coord_system)
                    except Exception as e:
                        print(f"Error determining coordinate system: {str(e)}")
                        self.detected_system_value.setText("Not detected")
                else:
                    self.status_bar.showMessage("No spherical cameras found for processing.")
                    self.detected_system_value.setText("Not detected")
            
            except Exception as e:
                QMessageBox.warning(self, "Error", f"{str(e)}")
        
        @staticmethod
        def show_window():
            global gui_window
            if gui_window is not None:
                gui_window.close()
            gui_window = CubemapConverterGUI()
            gui_window.show()
            return gui_window
                
        def start_processing(self):
            try:
                output_folder = self.output_folder_path.text()
                if output_folder == "Not selected" or not os.path.exists(output_folder):
                    QMessageBox.warning(self, "Error", "Please select a valid output folder.")
                    return
                
                selected_faces = [face for face, checkbox in self.face_checkboxes.items() if checkbox.isChecked()]
            
                if not selected_faces:
                    QMessageBox.warning(self, "Error", "Please select at least one face.")
                    return
            
                realign_cameras_after = self.realign_checkbox.isChecked()
                remove_spherical_cameras_after = self.remove_spherical_checkbox.isChecked()
                
                overlap = self.overlap_spinner.value()
                persp_size = self.size_combo.currentData()
                
                coord_system_option = self.coord_system_combo.currentText()
                if coord_system_option == "Auto-detect":
                    coord_system = determine_coordinate_system()
                else:
                    coord_system = coord_system_option
                
                format_text = self.format_combo.currentText()
                if "JPEG" in format_text:
                    file_format = "jpg"
                elif "PNG" in format_text:
                    file_format = "png"
                elif "TIFF" in format_text:
                    file_format = "tiff"
                else:
                    file_format = "jpg"
                
                quality = self.quality_spinner.value()
                interpolation = self.interp_combo.currentData()
                
                faces_threads = self.thread_spinner.value()
                
                doc = Metashape.app.document
                chunk = doc.chunk
                
                if not chunk:
                    QMessageBox.warning(self, "Error", "Active chunk not found.")
                    return
                
                spherical_cameras = [cam for cam in chunk.cameras if cam.transform and cam.photo]
                
                if not spherical_cameras:
                    QMessageBox.warning(self, "Error", "No spherical cameras found for processing.")
                    return
                
                options = {
                    "persp_size": persp_size,
                    "overlap": overlap,
                    "coord_system": coord_system,
                    "file_format": file_format,
                    "quality": quality,
                    "interpolation": interpolation,
                    "faces_threads": faces_threads,
                    "camera_threads": self.camera_thread_spinner.value(),
                    "selected_faces": selected_faces,
                    "realign_cameras_after": realign_cameras_after,
                    "remove_spherical_cameras_after": remove_spherical_cameras_after
                }

                self.process_thread = ProcessCamerasThread(
                    cameras=spherical_cameras,
                    output_folder=output_folder,
                    options=options
                )
                
                self.process_thread.update_progress.connect(self.update_progress)
                self.process_thread.processing_finished.connect(self.processing_finished)
                self.process_thread.error_occurred.connect(self.processing_error)
                
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.progress_bar.setValue(0)
                self.status_value.setText("Processing...")
                
                faces_threads = self.thread_spinner.value()
                camera_threads = self.camera_thread_spinner.value()
                self.status_bar.showMessage(f"Processing {len(spherical_cameras)} cameras using {camera_threads}/{faces_threads} threads...")

                self.process_thread.start()
                
                self.start_time = time.time()
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_remaining_time)
                self.timer.start(1000)
            
            except Exception as e:
                error_message = f"{str(e)}\n\n{traceback.format_exc()}"
                print(error_message)
                QMessageBox.critical(self, "Error", error_message)
        
        def stop_processing(self):
            if self.process_thread and self.process_thread.isRunning():
                reply = QMessageBox.question(
                    self, "Confirmation", 
                    "Are you sure you want to abort processing?", 
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.status_value.setText("Aborted")
                    self.status_bar.showMessage("Processing aborted by user")
                    self.process_thread.stop()
                    
        def update_progress(self, current, total, camera_name, status, progress_percent):
            self.progress_bar.setValue(progress_percent)
            self.current_camera_value.setText(camera_name)
            self.status_value.setText(status)
            self.status_bar.showMessage(f"Processing {total} cameras...")
        
        def update_remaining_time(self):
            if self.process_thread and self.process_thread.isRunning():
                elapsed = time.time() - self.start_time
                progress = self.progress_bar.value() / 100.0
                
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    
                    m, s = divmod(remaining, 60)
                    h, m = divmod(m, 60)
                    self.time_value.setText(f"{int(h):02d}:{int(m):02d}:{int(s):02d}")
        
        def processing_finished(self, success, stats):
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            if success:
                m, s = divmod(stats['time'], 60)
                h, m = divmod(m, 60)
                time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
                
                message = f"Processing completed!\n\n"
                message += f"Total cameras: {stats['total']}\n"
                message += f"Successfully processed: {stats['processed']}\n"
                message += f"Skipped/errors: {stats['skipped']}\n"
                message += f"Total time: {time_str}"
                
                if stats['skipped'] > 0 and 'errors' in stats and stats['errors']:
                    message += f"\n\nErrors ({len(stats['errors'])}):\n"
                    for i, error in enumerate(stats['errors'][:5]):
                        message += f"{i+1}. {error}\n"
                    if len(stats['errors']) > 5:
                        message += f"... and {len(stats['errors']) - 5} more errors."
                
                QMessageBox.information(self, "Completed", message)
                self.status_value.setText("Completed")
                self.status_bar.showMessage(f"Processing complete. Success: {stats['processed']}, Errors: {stats['skipped']}")
            else:
                message = "Processing aborted by user."
                QMessageBox.warning(self, "Aborted", message)
                self.status_value.setText("Aborted")
                self.status_bar.showMessage("Processing aborted by user")
        
        def processing_error(self, error_message):
            QMessageBox.critical(self, "Error", f"{error_message}")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_value.setText("Error")
            self.status_bar.showMessage("Processing error")
            
        def closeEvent(self, event):
            if self.process_thread and self.process_thread.isRunning():
                reply = QMessageBox.question(
                    self, "Confirmation", 
                    "Processing is not complete. Are you sure you want to exit?", 
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.process_thread.stop()
                    self.process_thread.wait(1000)
                    event.accept()
                else:
                    event.ignore()
# === CONSOLE & MAIN ===

def process_images_console():
    """
    Console version of image processing function.
    Uses multithreading for faster processing.
    """
    try:
        print("Checking active chunk...")
        doc = Metashape.app.document
        if not doc:
            Metashape.app.messageBox("Error: No active chunk found.")
            return
            
        chunk = doc.chunk
        if not chunk:
            Metashape.app.messageBox("Error: No active chunk found.")
            return
            
        print("Checking spherical cameras...")
        spherical_cameras = [cam for cam in chunk.cameras if cam.transform and cam.photo]
        
        if not spherical_cameras:
            Metashape.app.messageBox("Error: No spherical cameras found for processing.")
            return
            
        print(f"Found {len(spherical_cameras)} spherical cameras.")
            
        output_folder = Metashape.app.getExistingDirectory("Select output folder for saving images")
        if not output_folder:
            return
        
        output_folder = normalize_path(output_folder)
            
        overlap = Metashape.app.getFloat("Enter overlap value (0-20)", 10.0)
        if overlap is None or overlap < 0 or overlap > 20:
            Metashape.app.messageBox("Error: Overlap value should be between 0 and 20.")
            return
            
        size_options = ["Automatically (recommended)", "512x512", "1024x1024", "2048x2048", "4096x4096"]
        selected_size = get_string_option("Select cube face size", size_options)
        
        persp_size = None
        if selected_size != "Automatically (recommended)":
            persp_size = int(selected_size.split("x")[0])
        
        format_options = ["jpg", "png", "tiff"]
        file_format = get_string_option("Select file format (jpg, png, tiff)", format_options)
        
        quality = Metashape.app.getInt("Enter image quality (75-100)", 95, 75, 100)
        
        interp_options = ["Nearest", "Linear", "Cubic"]
        interp_option = get_string_option("Select interpolation method", interp_options)
        
        interpolation = cv2.INTER_CUBIC
        if interp_option == "Nearest":
            interpolation = cv2.INTER_NEAREST
        elif interp_option == "Linear":
            interpolation = cv2.INTER_LINEAR
        
        face_options = ["front", "right", "left", "top", "down", "back"]
        selected_faces = []
        
        for face in face_options:
            if Metashape.app.getBool(f"Generate {face} face?", True):
                selected_faces.append(face)
        
        if not selected_faces:
            Metashape.app.messageBox("Error: Please select at least one face.")
            return
            
        realign_cameras_after = Metashape.app.getBool("Re-align cameras after conversion?", False)
        remove_spherical_cameras_after = Metashape.app.getBool("Remove source spherical cameras?", False)
            
        coord_system = determine_coordinate_system()
        print(f"Determined coordinate system: {coord_system}")
        
        cpu_count_local = os.cpu_count() or 1
        recommended_face_threads = min(6, max(1, cpu_count_local // 2))
        face_threads = Metashape.app.getInt(f"Number of face threads (recommended {recommended_face_threads}):",
                                          recommended_face_threads, 1, max(1, cpu_count_local // 2))

        recommended_camera_threads = cpu_count_local
        camera_threads = Metashape.app.getInt(f"Number of camera threads (recommended {recommended_camera_threads}):",
                                            recommended_camera_threads, 1, cpu_count_local)

        print(f"\nStarting processing {len(spherical_cameras)} cameras ({camera_threads} camera threads / {face_threads} face threads)...")
        print(f"Settings: overlap={overlap}, face size={selected_size}, system={coord_system}")
        print(f"Format={file_format}, quality={quality}, interpolation={interp_option}")
        print(f"Selected faces: {', '.join(selected_faces)}")

        start_time = time.time()
        processed_count = 0
        skipped_count = 0
        add_camera_errors_console = []
        conversion_results_console = []
        futures_console = []

        info_message = f"Starting processing {len(spherical_cameras)} cameras.\n\nSettings:\n- Overlap: {overlap} degrees\n- Face size: {selected_size}\n- Coordinate system: {coord_system}\n- File format: {file_format}\n- Quality: {quality}\n- Interpolation: {interp_option}\n- Threads: {camera_threads}/{face_threads}\n\nResults will be saved to:\n{output_folder}"
        Metashape.app.messageBox(info_message)

        print(f"--- Stage 1: Image Conversion ({camera_threads} threads) ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=camera_threads) as executor:
            def _process_single_camera_console(camera_idx, cam):
                try:
                    image_paths_c = convert_spherical_to_cubemap(
                        spherical_image_path=cam.photo.path,
                        output_folder=output_folder,
                        camera_label=cam.label,
                        persp_size=persp_size,
                        overlap=overlap,
                        file_format=file_format,
                        quality=quality,
                        interpolation=interpolation,
                        max_workers=face_threads,
                        selected_faces=selected_faces
                    )
                    if image_paths_c:
                        first_img = list(image_paths_c.values())[0]
                        img = read_image_safe(first_img)
                        actual_size_c = img.shape[0] if img is not None else None
                        if actual_size_c is None:
                             print(f"\nWarning: Failed to read {first_img} for camera {cam.label}")
                             return {"camera": cam, "image_paths": image_paths_c, "actual_size": None, "error": f"Read error for {first_img}"}
                        else:
                             return {"camera": cam, "image_paths": image_paths_c, "actual_size": actual_size_c, "error": None}
                    else:
                        return {"camera": cam, "image_paths": None, "actual_size": None, "error": "skipped_no_faces"}
                except Exception as e:
                    return {"camera": cam, "image_paths": None, "actual_size": None, "error": str(e)}

            for i, camera in enumerate(spherical_cameras):
                 future = executor.submit(_process_single_camera_console, i, camera)
                 futures_console.append(future)

            conversion_completed_count_console = 0
            for future in concurrent.futures.as_completed(futures_console):
                result = future.result()
                conversion_completed_count_console += 1
                camera_label_c = result["camera"].label
                if result["error"]:
                    skipped_count += 1
                    if result["error"] != "skipped_no_faces":
                         print(f"\nError processing camera {camera_label_c}: {result['error']}")
                else:
                    conversion_results_console.append(result)

                console_progress_bar(conversion_completed_count_console, len(spherical_cameras), prefix="[Conversion] ", suffix=f"{conversion_completed_count_console}/{len(spherical_cameras)}", length=40)

        console_progress_bar(len(spherical_cameras), len(spherical_cameras), prefix="[Conversion] ", suffix=" Complete", length=40)
        print()

        print(f"--- Stage 2: Adding cameras to Metashape (Sequential) ---")
        total_to_add_console = len(conversion_results_console)
        added_count_console = 0
        if total_to_add_console > 0:
            chunk_c = Metashape.app.document.chunk
            for result in conversion_results_console:
                added_count_console += 1
                console_progress_bar(added_count_console, total_to_add_console, prefix=f"[Adding cameras {added_count_console}/{total_to_add_console}] ", suffix=f"{result['camera'].label}", length=40)
                try:
                    if result["actual_size"] is None:
                        raise ValueError("Actual size is None, cannot add camera.")
                    add_cubemap_cameras(
                        chunk=chunk_c,
                        spherical_camera=result["camera"],
                        image_paths=result["image_paths"],
                        persp_size=result["actual_size"],
                        coord_system=coord_system
                    )
                    processed_count += 1
                except Exception as e:
                     error_msg_add = f"Error adding camera {result['camera'].label}: {str(e)}"
                     print(f"\n{error_msg_add}")
                     add_camera_errors_console.append(error_msg_add)

            console_progress_bar(total_to_add_console, total_to_add_console, prefix=f"[Adding cameras {total_to_add_console}/{total_to_add_console}] ", suffix=" Complete", length=40)
            print()

        print("--- Stage 3: Post-processing ---")
        if realign_cameras_after:
            try:
                print("Re-aligning cameras...")
                realign_cameras()
            except Exception as e:
                print(f"Error re-aligning cameras: {str(e)}")

        if remove_spherical_cameras_after:
            try:
                print("Removing spherical cameras...")
                remove_spherical_cameras()
            except Exception as e:
                 print(f"Error removing spherical cameras: {str(e)}")

        end_time_total = time.time()
        final_skipped_console = skipped_count + (total_to_add_console - processed_count)
        print("\n==== Processing Results ====")
        print(f"Total cameras: {len(spherical_cameras)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped/errors: {final_skipped_console}")
        if add_camera_errors_console:
            print(f"Errors ({len(add_camera_errors_console)}):")
            for err in add_camera_errors_console:
                print(f"- {err}")
        
        m, s = divmod(end_time_total - start_time, 60)
        h, m = divmod(m, 60)
        print(f"Total time: {int(h):02d}:{int(m):02d}:{int(s):02d}")

    except Exception as e:
        error_message = f"{str(e)}\n\n{traceback.format_exc()}"
        print(error_message)
        try:
            Metashape.app.messageBox(f"General processing error: {str(e)}")
        except:
            print(f"Error showing message box: {str(e)}")

gui_window = None

def main():
    """
    Main entry point.
    """
    global gui_window

    setup_locale_and_encoding()
    fix_metashape_file_paths()

    print("--- Spherical to Cubemap Converter ---")
    print("Version 1.0.1 - Optimized")

    if use_gui:
        try:
            print("Initializing graphical interface...")
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            gui_window = CubemapConverterGUI()
            gui_window.show()
            
            print("Graphical interface launched successfully")

            try:
                non_blocking = True
                if non_blocking:
                    print("Non-blocking mode selected")
                else:
                     print("Blocking mode selected")
                     print("Starting Qt event processing loop...")
                     app.exec_()
                     print("Qt event processing loop completed")

            except AttributeError:
                print("Running outside Metashape")
                print("Starting Qt event processing loop...")
                app.exec_()
                print("Qt event processing loop completed")
            except Exception as e:
                 print(f"Error querying mode: {str(e)}")
                 print("Defaulting to non-blocking mode")

        except Exception as e:
            print(f"GUI initialization error: {str(e)}")
            print(traceback.format_exc())
            print("Switching to console mode.")
            process_images_console()
    else:
        process_images_console()

if __name__ == "__main__":
    main()
