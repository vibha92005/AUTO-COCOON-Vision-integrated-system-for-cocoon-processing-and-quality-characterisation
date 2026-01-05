# YOLOv8 Integration for Real-Time Cocoon Classification

## Overview
This module handles the real-time integration of the trained YOLOv8 cocoon classification model with incoming images. It continuously monitors a specified folder for new cocoon images, performs inference using the trained model, and classifies cocoons as **Good** or **Bad**.

The integration workflow is designed to support automated or semi-automated cocoon sorting systems and can be extended to hardware-based actuation (servo motors, conveyors, etc.).

---

## System Workflow
1. Images are captured using an external camera setup (Intel Depth Camera or phone camera).
2. Captured images are saved to a shared folder (Google Drive).
3. The integration script continuously monitors this folder.
4. When a new image is detected:
   - YOLOv8 performs inference.
   - Detected classes are analyzed.
   - Cocoon quality is classified as **Good** or **Bad**.
5. The annotated output image is saved and displayed.

---

## Folder Structure
integration/
│
├── integration.py          # Real-time inference script
├── README.md               # This file
└── output/                 # Annotated inference results

---

## Requirements
- Python 3.8 or higher  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  
- Google Drive (for image synchronization)

Install dependencies using:
pip install ultralytics opencv-python numpy

---

## Model Configuration
The integration uses the trained YOLOv8 model:

- Model file: `best_final_push.pt`
- Classes:
  - `good`
  - `bad`

The model is loaded directly from Google Drive for ease of deployment.

---

## Integration Logic
- The script checks the image folder at regular intervals.
- Only new images (not previously processed) are selected.
- Inference is performed using YOLOv8.
- If the class **good** is detected → cocoon is classified as GOOD.
- Otherwise → cocoon is classified as BAD.
- Annotated images with bounding boxes are saved automatically.

---

## Output
- Console-based classification result (GOOD / BAD)
- Annotated image showing detected cocoon class
- Ready for extension to hardware-based sorting mechanisms

---

## Notes
- Supported image formats: `.jpg`, `.jpeg`, `.png`
- Image filenames must be unique to avoid reprocessing
- The folder-based approach allows easy integration with external camera systems
- Hardware control logic (servo, relay, etc.) can be added after classification
