# Spherical to Cubemap Converter for Agisoft Metashape

Converts spherical (equirectangular) images into cubemap faces for photogrammetry in Agisoft Metashape Pro.

## Scripts

| Script | Description |
|--------|-------------|
| `unified_fixed_v002.py` | **Recommended.** All-in-one converter with COLMAP export for 3D Gaussian Splatting. No realignment needed. |
| `convert_to_cubemap_v012.py` | Standard 6-face cubemap converter with GUI. |
| `convert_to_cubemap_v013_high_overlap.py` | 10-face "High Overlap" mode (8 horizontal + Top + Bottom) for better reconstruction. |

---

## Quick Start (Recommended Workflow)

1. Import spherical images into Metashape
2. Align cameras (once)
3. Run `unified_fixed_v002.py` via `Tools → Run Script`
4. Done! COLMAP export folder is created automatically

**No realignment needed** after running the script.

---

## Output Structure (unified_fixed_v002.py)

```
output_folder/
├── images/           # Cubemap face images
├── sparse/0/         # COLMAP data for 3DGS
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── README_FIXED.txt
```

---

## Requirements

- **Agisoft Metashape Pro** (v1.6+)
- **Python libraries** (auto-installed on first run):
  - `opencv-python`
  - `PyQt5`

---

## GUI Options (v012 / v013)

If using the GUI scripts:

- **Output folder**: Where to save generated images
- **Overlap**: Angle overlap between faces (degrees)
- **Cube face size**: Resolution per face ("Auto" = source width ÷ 4)
- **Layout** (v013 only): Standard (6) or High Overlap (10 faces)
- **File format**: JPG, PNG, or TIFF
- **Threading**: Adjust for your RAM (see below)

### Threading Tips

| Setting | Recommendation |
|---------|----------------|
| **Camera threads = 1** | Safe for 16-32GB RAM |
| **Face threads = 4-6** | Good balance of speed/stability |
| **High RAM (64GB+)** | Can increase camera threads |

---

## Troubleshooting

- **Crashes/High RAM**: Set "Camera threads" to 1
- **Missing libraries**: Install manually: `pip install opencv-python PyQt5`
- **Path issues**: Avoid special characters in file paths

---

## License

See [LICENSE.md](LICENSE.md)
