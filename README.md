# AUTO-COCOON: Vision-Integrated System for Cocoon Processing and Quality Characterisation

## Overview
AUTO-COCOON is an AI-driven vision-based system designed for automated cocoon processing and quality characterisation. The project integrates **deep learning–based object detection**, **image enhancement using a Convolutional Autoencoder (CAE)**, and **real-time inference** to classify silk cocoons as *Good* or *Bad*.

The system focuses on improving dataset quality, increasing classification accuracy, and enabling scalable cocoon sorting using computer vision techniques.

---

## Key Objectives
- Automate cocoon quality assessment using AI  
- Enhance training data using CAE-based image augmentation  
- Train an accurate YOLOv8 model for cocoon classification  
- Enable real-time inference through system integration  

---

## System Pipeline
1. Image Acquisition  
   Cocoon images are captured using an Intel Depth Camera or external camera source.

2. Image Enhancement & Augmentation (CAE)  
   A Convolutional Autoencoder generates visually safe augmented images to improve model generalization.

3. YOLOv8 Training
   The augmented dataset is used to train a YOLOv8 model for cocoon quality classification.

4. Real-Time Integration 
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
|
|-- cae_augmentation/          # CAE-based image enhancement & augmentation
|   |-- cae_augmentation.py
|   |-- README.md
|
|-- yolo_training/             # YOLOv8 training and evaluation
|   |-- train.py
|   |-- README.md
|
|-- integration/               # Real-time inference and system integration
|   |-- integration.py
|   |-- README.md
|
|-- .gitignore
|-- LICENSE
|-- requirements.txt
|-- README.md                  # Project overview (this file)

---


---

## Features
- CAE-based dataset augmentation with SSIM-preserving loss  
- YOLOv8-based cocoon quality classification  
- Real-time image monitoring and inference  
- Modular design for future hardware-based sorting integration  
- Clean, industry-ready project structure  

---

## Results
- Improved dataset diversity using CAE augmentation  
- Accurate classification of cocoons into *Good* and *Bad* categories  
- Successful real-time inference using folder-based integration  

---

## Future Enhancements
- Servo motor or conveyor-based automated sorting  
- Edge deployment (Jetson Nano / Raspberry Pi)  
- Cloud dashboard for monitoring and analytics  
- Multi-grade cocoon classification  

---

## Author
**Vibha I S**  
B.E. Electronics and Communication Engineering  

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

