# ------------------------------
# YOLOv8 Training & Validation
# ------------------------------

#  Mount Google Drive
from google.colab import drive
import os
from ultralytics import YOLO

drive.mount('/content/gdrive')

#  Paths
ZIP_PATH = '/content/gdrive/MyDrive/finally.zip'
YAML_PATH = '/content/finally/data.yaml'
NEW_ROOT_PATH = '/content/finally/'
MODEL_PATH = '/content/gdrive/MyDrive/best_final_push.pt'

#  Unzip dataset
!unzip -q "{ZIP_PATH}" -d .

#  Fix data.yaml paths
if not os.path.exists(YAML_PATH):
    raise FileNotFoundError(f"YAML file not found at {YAML_PATH}")
else:
    print(f"Resetting paths in {YAML_PATH}...")

    with open(YAML_PATH, 'r') as f:
        yaml_content = f.read()

    corrected_lines = []

    for line in yaml_content.splitlines():
        line_stripped = line.strip()

        if line_stripped.startswith('path:'):
            corrected_lines.append(f"path: {NEW_ROOT_PATH}")
            print("  Fixed 'path:' line")
        elif line_stripped.startswith('train:'):
            corrected_lines.append("train: train/images")
            print("  Fixed 'train:' to 'train/images'")
        elif line_stripped.startswith('val:'):
            corrected_lines.append("val: val/images")
            print("  Fixed 'val:' to 'val/images'")
        elif line_stripped.startswith('test:'):
            corrected_lines.append("test: test/images")
            print("  Fixed 'test:' to 'test/images'")
        else:
            corrected_lines.append(line)

    with open(YAML_PATH, 'w') as f:
        f.write('\n'.join(corrected_lines))

    print("\n YAML file paths fixed. Ready for training.")

# Load the trained YOLOv8 model
print(f"\nLoading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Run validation/evaluation
print(f"\n--- Running Validation on Best Model using {YAML_PATH} ---")
metrics = model.val(data=YAML_PATH)
print("\n Validation completed.")
