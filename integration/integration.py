"""
AUTO-COCOON: Real-Time Cocoon Sorting Integration
-------------------------------------------------
This script watches a folder for new cocoon images and classifies them
as GOOD or BAD using a trained YOLOv8 model. Results are displayed
and saved with annotations.
"""

# ------------------------------
# Install & Imports
# ------------------------------
!pip install -q ultralytics

import os
import time
from ultralytics import YOLO
from IPython.display import display, Image
from google.colab import drive

# ------------------------------
# Mount Google Drive
# ------------------------------
drive.mount('/content/drive')
print("✅ Drive mounted!")

# ------------------------------
# Paths
# ------------------------------
MODEL_PATH = "/content/drive/MyDrive/best_final_push.pt"
WATCH_FOLDER = "/content/drive/MyDrive/CocoonImages"
OUTPUT_FOLDER = "/content/drive/MyDrive/CocoonOutput"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------
# Load YOLO Model
# ------------------------------
print(f"📦 Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------------------
# Processed files tracker
# ------------------------------
processed_files = set()

print(f"👀 Watching folder: {WATCH_FOLDER} for new images...")

# ------------------------------
# Main Loop
# ------------------------------
try:
    while True:
        files = sorted(os.listdir(WATCH_FOLDER))
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")) and file_name not in processed_files:
                full_path = os.path.join(WATCH_FOLDER, file_name)
                print(f"\n📸 New Image Detected: {file_name}")

                # Run inference
                results = model(full_path)
                pred = results[0]

                # Determine GOOD / BAD
                detected_classes = [model.names[int(box.cls)] for box in pred.boxes]

                if "good" in detected_classes:
                    print("🟢 RESULT: GOOD COCOON")
                else:
                    print("🔴 RESULT: BAD COCOON")

                # Save annotated image
                output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{file_name}")
                pred.plot(save=True, filename=output_path)

                # Display annotated image in Colab
                display(Image(filename=output_path))

                # Mark as processed
                processed_files.add(file_name)

        # Wait before checking again
        time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Stopped watching folder.")
