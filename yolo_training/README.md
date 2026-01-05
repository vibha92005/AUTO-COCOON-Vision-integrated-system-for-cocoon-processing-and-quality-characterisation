# YOLOv8 Training for Cocoon Quality Classification

## Overview
This module implements the training, validation, and evaluation of a YOLOv8 object detection model for cocoon quality classification (Good and Bad cocoons). The model is trained using an augmented dataset generated through a Convolutional Autoencoder (CAE) to improve robustness and generalization.

The trained YOLO model is later used for real-time cocoon classification and automated sorting.

---

## Dataset Structure
The dataset follows the standard YOLO directory format:

dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml

---

## Annotation Format
Each label file follows YOLO format:

<class_id> <x_center> <y_center> <width> <height>

All values are normalized between 0 and 1.

---

## data.yaml Configuration
Example `data.yaml` file:

path: /content/finally  
train: train/images  
val: val/images  
test: test/images  

nc: 2  
names: ['good', 'bad']

---

## Requirements
- Python 3.8 or higher  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  

Install YOLOv8 using:
pip install ultralytics

---

## Training
YOLOv8 is fine-tuned using a pretrained model on the cocoon dataset.

Example training command:
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640

The best model weights are automatically saved as:
runs/detect/train/weights/best.pt

---

## Evaluation
Model evaluation is performed using:
yolo detect val model=best.pt data=data.yaml

Evaluation metrics include:
- Precision  
- Recall  
- mAP@0.5  
- mAP@0.5:0.95  

---

## Output
- Trained model weights (best.pt)  
- Training logs and performance plots  
- Validation and test metrics  

The trained model is used in the real-time inference and integration module.

---

## Notes
- Image and label filenames must match exactly.  
- Augmented images generated using the CAE module can be directly included in the training dataset.  
- Dataset quality and class balance significantly impact detection performance.
