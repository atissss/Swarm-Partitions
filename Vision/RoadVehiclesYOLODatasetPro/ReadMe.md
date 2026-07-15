# Merged and Modified Aerial Traffic, Roundabouts Traffic (Aerial/Drone View), VSAI & VisDrone Dataset (YOLO Format)

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![YOLO Format](https://img.shields.io/badge/Format-YOLO-%2344aa11)](https://ultralytics.com)

A merged YOLO-formatted dataset combining **[Aerial Traffic Images](https://www.kaggle.com/datasets/cihangiryiit/aerial-traffic-images)**, **[VisDrone](https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset)**, **[Roundabout Aerial Images for Vehicle Detection](https://www.kaggle.com/datasets/javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection)** and **[VSAI_Dataset](https://www.kaggle.com/datasets/dronevision/vsaiv1)**, with modifications for enhanced usability with ultralytics' YOLOv8/11 for vehicle detection in aerial/drone view tasks.

---

## Dataset Overview
- **Purpose**: Training and benchmarking object detection models (YOLOv8/v11) for aerial surveillance, traffic monitoring, and drone applications.
- **Key Features**:
  - Merged classes from all four datasets into three classes: small-vehicle, large-vehicle, and human.
  - Preprocessed and converted to YOLO format (normalized annotations, standardized splits).
  - Resolved label conflicts and improved annotation consistency.

---

## Dataset Details
### Sources
1. **Aerial Traffic Images**: Focuses on aerial traffic scenes with vehicles and pedestrians.
2. **VisDrone Dataset**: Captures diverse objects (cars, pedestrians, etc.) in complex environments.
3. **Roundabout Aerial Images for Vehicle Detection**: Provides roundabout-specific aerial images for vehicle detection.
4. **VSAI_Dataset**: Contains aerial imagery for traffic analysis by DroneVision.

### Modifications
- **Merging Process**:
  - Simplified the classes into three classes (small-vehicle, large-vehicle, human) for focused traffic detection.
- **Preprocessing**:
  - Removed corrupted or duplicate annotations.
  - Split into train/val/test sets (70%/15%/15%).
- **YOLO Conversion**:
  - Labels converted to `.txt` files with normalized coordinates.
  - Structured directory hierarchy for compatibility with YOLO pipelines.

### Classes
| Class ID | Class Name    | Source Dataset                                   |
|----------|---------------|--------------------------------------------------|
| 0        | small-vehicle | Aerial Traffic, VisDrone, Roundabout, VSAI       |
| 1        | large-vehicle | Aerial Traffic, VisDrone, Roundabout, VSAI       |
| 2        | human         | VisDrone      |

**Total Classes**: 3

### Statistics
- **Total Images**: 35,886  
  - Train: 25,120
  - Val: 5,384
  - Test: 5,382
- **Annotations**: ~1,200,000 instances

---

## 📁 Dataset Structure

    RoadVehiclesYOLODatasetPro/ 
    ├── train/ 
    │ 
    ├── images/ # 25120 images 
    │ 
    └── labels/ # 25120 labels 
    ├── test/ 
    │ 
    ├── images/ # 5384 images 
    │ 
    └── labels/ # 5384 labels 
    ├── val/ 
    │ 
    ├── images/ # 5382 images 
    │ 
    └── labels/ # 5382 labels 
    ├── data.yaml # The YOLO dataset configuration file. 
    ├── license.md # The license file. 
    └── ReadMe.md # This documentation file.

### Explanation:
- **`train/`, `val/` & `test/`**: Contain the split datasets for model training and evaluation (with K-fold cross validation).
- **`images/`**: Contains input images.
- **`labels/`**: Contains YOLO-format `.txt` files with bounding box annotations (format: class x_center y_center width height).
- **`data.yaml`**: Defines dataset paths and class names.

---

### Example `data.yaml`:
```yaml
# Dataset configuration
path: RoadVehiclesYOLODatasetPro/
train: train/images
val: val/images
test: test/images

nc: 3
# Class names
names:
  0: small-vehicle
  1: large-vehicle
  2: human



