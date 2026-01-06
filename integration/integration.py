# Real-Time Cocoon Classification Integration

## Overview
This module implements the real-time integration pipeline for cocoon
classification using a trained YOLOv8 model.

The system continuously monitors a Google Drive folder for newly captured
cocoon images, performs inference, and classifies each cocoon as
**Good** or **Bad**. Annotated results are saved and displayed for
verification.

This module focuses on **vision-based decision output only** and does not
include physical actuation or sorting hardware.

---

## Integration Workflow
1. Image Source  
   Cocoon images are captured externally and uploaded to a designated
   Google Drive folder.

2. Folder Monitoring  
   The system continuously watches the folder for newly added images.

3. YOLOv8 Inference  
   Each new image is passed through a trained YOLOv8 model for cocoon
   quality classification.

4. Result Generation  
   - Detected cocoons are classified as **Good** or **Bad**
   - Bounding boxes and labels are drawn on the image
   - Annotated images are saved for reference

5. Visualization  
   Annotated outputs are displayed in real time during execution.

---

## Directory Usage
- Input Folder  
  Google Drive directory containing newly uploaded cocoon images.

- Output Folder  
  Automatically generated folder storing annotated inference results.

---

## Implementation Highlights
- Google Drive is used for cloud-based image synchronization.
- Folder-based polling enables near real-time processing.
- YOLOv8 (Ultralytics) is used for detection and classification.
- Processed images are tracked to avoid duplicate inference.

---

## Limitations
- Physical sorting using servo motors or conveyors is **not implemented**
  in the current version.
- The system outputs classification results only.

---

## Future Enhancements
- Integration with servo motors for automated cocoon sorting.
- Edge deployment on embedded platforms (Jetson Nano / Raspberry Pi).
- Web-based dashboard for live monitoring and analytics.
