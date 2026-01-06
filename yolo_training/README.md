YOLOv8 Training for Cocoon Quality Classification

Overview

This module implements the training, validation, and evaluation of a YOLOv8 object detection model for cocoon quality classification into Good and Bad categories.
The model is trained using a dataset enhanced through Convolutional Autoencoder (CAE)–based augmentation to improve robustness, generalization, and detection accuracy.
The trained model is later used for real-time cocoon classification in the integration module.

Dataset Structure

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


Annotation Format
Each image has a corresponding label file in YOLO format:

<class_id> <x_center> <y_center> <width> <height>


All values are normalized between 0 and 1
Class IDs:
0 → good
1 → bad

data.yaml Configuration
Example configuration used during training:

path: /content/finally
train: train/images
val: val/images
test: test/images
nc: 2
names: ['good', 'bad']

Requirements
Python 3.8 or higher
Ultralytics YOLOv8
OpenCV
NumPy

Install YOLOv8 using:
pip install ultralytics

Training Procedure
YOLOv8 is fine-tuned using pretrained weights on the cocoon dataset.

Example training command:
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640


During training:
Training loss and metrics are logged automatically

The best-performing weights are saved as:
runs/detect/train/weights/best.pt

Evaluation
Model evaluation is performed using:
yolo detect val model=best.pt data=data.yaml

Evaluation metrics include:
Precision
Recall
mAP@0.5
mAP@0.5:0.95

Outputs
Trained model weights (best.pt)
Training logs and performance plots
Validation and test metrics

The integration module uses the trained model for real-time cocoon classification.

Notes
Image and label filenames must match exactly
Augmented images generated using the CAE module can be directly included in the training dataset
Dataset quality and class balance significantly impact model performance
This module focuses only on model training and evaluation

Module Status
Training: Implemented
Evaluation: Implemented
CAE-Augmented Dataset: Supported
Physical Sorting: Not implemented
