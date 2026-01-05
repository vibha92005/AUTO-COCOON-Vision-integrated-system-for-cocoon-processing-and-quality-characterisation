# AUTO-COCOON: Vision-Integrated System for Cocoon Processing and Quality Characterisation

## Overview
AUTO-COCOON is an AI-driven vision-based system designed for automated cocoon processing and quality characterisation. The project integrates **deep learning–based object detection**, **image enhancement using a Convolutional Autoencoder (CAE)**, and **real-time inference** to classify silk cocoons as *Good* or *Bad*.

The system aims to reduce manual inspection, improve consistency, and enable scalable cocoon sorting using computer vision and AI techniques.

---

## Key Objectives
- Automate cocoon quality assessment using deep learning  
- Improve training data quality through CAE-based image augmentation  
- Enable real-time cocoon classification using YOLOv8  
- Provide a modular and extensible pipeline suitable for industrial automation  

---

## System Architecture
1. **Image Acquisition**  
   Cocoon images are captured using an Intel Depth Camera or external camera setup.

2. **Image Enhancement & Augmentation (CAE)**  
   A Convolutional Autoencoder generates visually-safe augmented images to improve model generalization.

3. **YOLOv8 Training**  
   The augmented dataset is used to train a YOLOv8 model for cocoon quality classification.

4. **Real-Time Integration**  
   The trained model performs inference on newly captured images and classifies cocoons as *Good* or *Bad*.

---

## Technologies Used
- Python  
- YOLOv8 (Ultralytics)  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Google Colab & Google Drive  
- Intel Depth Camera  

---

## Repository Structure
AUTO-COCOON/
│
├── cae_augmentation/        # CAE-based image enhancement & augmentation
│   ├── cae_augmentation.py
│   └── README.md
│
├── yolo_training/           # YOLOv8 training and evaluation
│   ├── train.py
│   └── README.md
│
├── integration/             # Real-time inference and system integration
│   ├── integration.py
│   └── README.md
│
├── README.md                # Project overview (this file)
└── LICENSE

---

## Features
- CAE-based dataset augmentation with SSIM-preserving loss  
- YOLOv8-based cocoon quality classification  
- Real-time image monitoring and inference  
- Modular design for easy extension to hardware-based sorting  
- Industry-ready AI workflow  

---

## Results
- Improved dataset diversity through CAE augmentation  
- Accurate classification of cocoons into Good and Bad categories  
- Successful real-time inference using folder-based image monitoring  

---

## Future Enhancements
- Integration with servo motors or conveyor systems for automated sorting  
- Deployment on edge devices (Jetson Nano / Raspberry Pi)  
- Cloud-based dashboard for monitoring and analytics  
- Multi-class cocoon grading  

---

## Author
**Vibha I S**  
B.E. Electronics and Communication Engineering  

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
