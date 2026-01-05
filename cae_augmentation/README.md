# Convolutional Autoencoder (CAE) for Cocoon Image Augmentation

## Overview
This module implements a **Convolutional Autoencoder (CAE)** for augmenting cocoon images to improve the training dataset for YOLO-based cocoon classification. The CAE enhances images while preserving structural features, adding controlled noise and subtle variations for better model generalization.

The system uses a **U-Net style architecture** with weakened skip connections and a **custom combined loss** of Mean Squared Error (MSE) and Structural Similarity Index (SSIM) for visually-safe augmentations.

## Features
- Automated augmentation of cocoon images.  
- Controlled noise (`noise_std = 0.07`) to maintain visual quality.  
- SSIM-based loss (`ssim_weight = 0.75`) ensures structural similarity.  
- Sequentially saved augmented images and corresponding labels.  
- Compatible with YOLO training workflow.  
- Visualization of augmented vs original images for verification.

## Folder Structure
cae_augmentation/
│
├── cae_augmentation.py # Main CAE code
├── README.md # This file
├── output/ # Generated images and labels
│ ├── images/
│ └── labels/


## Requirements
- Python 3.8+  
- TensorFlow 2.x  
- OpenCV  
- NumPy  
- scikit-image  
- Matplotlib  
- scikit-learn  

Install dependencies via:
      pip install tensorflow opencv-python numpy scikit-image matplotlib scikit-learn

Usage

1.Mount your Google Drive in Colab (optional)
    from google.colab import drive
    drive.mount('/content/drive')

2.Set the paths for source images and output directory:
    base_dir = "/content/tiget"          # Original cocoon images
    output_dir = "/content/drive/MyDrive/cae_results"

3.Run the CAE code:
    !python cae_augmentation.py

Augmented images will be saved in output/images/ and labels in output/labels/ with sequential numbering.

Notes
1.Make sure your original images follow the color_001.png naming format.
2.The code automatically handles train/validation splits and sequentially numbers augmented images.
3.Use the augmented dataset for YOLOv8 training.
