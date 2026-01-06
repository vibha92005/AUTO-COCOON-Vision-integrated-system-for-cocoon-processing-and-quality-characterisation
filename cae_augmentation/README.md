**Convolutional Autoencoder (CAE) for Cocoon Image Augmentation**

**Overview**

This module implements a Convolutional Autoencoder (CAE)–based image augmentation pipeline to enhance cocoon image datasets for training deep learning models such as YOLOv8.

The CAE generates visually realistic augmented cocoon images while preserving key structural features. Controlled noise and feature reconstruction improve dataset diversity, reduce overfitting, and enhance generalization of the downstream cocoon quality classification model.

A U-Net–inspired architecture with weakened skip connections is used, along with a custom composite loss function combining Mean Squared Error (MSE) and Structural Similarity Index (SSIM) to ensure visually safe augmentations.

**Key Features**

Automated cocoon image augmentation using CAE
Controlled Gaussian noise (noise_std = 0.07) for realistic variation
SSIM-weighted loss (ssim_weight = 0.75) to preserve structure
Generates augmented images and corresponding YOLO-compatible labels
Sequential image naming to avoid dataset conflicts
Visual comparison of original and augmented images for validation
Fully compatible with YOLOv8 training pipeline

**Folder Structure**
cae_augmentation/
│
├── cae_augmentation.py      # Main CAE implementation
├── README.md                # Documentation
├── output/
│   ├── images/              # Augmented images
│   └── labels/              # Corresponding YOLO labels



**Requirements**

Python 3.8 or higher
TensorFlow 2.x
OpenCV
NumPy
scikit-image
Matplotlib
scikit-learn

**Install dependencies**
pip install tensorflow opencv-python numpy scikit-image matplotlib scikit-learn

**Usage**
1. (Optional) Mount Google Drive in Google Colab
from google.colab import drive
drive.mount('/content/drive')

2. Configure dataset paths

Inside cae_augmentation.py, set:

base_dir = "/content/tiget"                 # Original cocoon images
output_dir = "/content/drive/MyDrive/cae_results"

3. Run the CAE augmentation
python cae_augmentation.py

**Output**

Augmented cocoon images saved in:
output/images/
Corresponding YOLO-format label files saved in:

output/labels/
Images are sequentially numbered to avoid overwriting
Visual comparison plots are generated for verification

**Notes**
Original images should follow a consistent naming format (e.g., color_001.png).
The code automatically handles dataset splitting and sequential numbering.
Augmented images can be directly merged into the YOLO training dataset.
Dataset quality and balance significantly influence final model performance.

**Module Status**
CAE Model: Implemented
Image Augmentation: Implemented
YOLO Compatibility: Supported
Physical Sorting / Actuation: Not implemented
