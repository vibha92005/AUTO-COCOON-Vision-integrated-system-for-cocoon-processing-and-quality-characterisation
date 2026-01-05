"""
AUTO-COCOON: Convolutional Autoencoder (CAE) for Cocoon Image Augmentation
-------------------------------------------------------------------------

This script performs data augmentation on cocoon images using a Convolutional Autoencoder (CAE)
with controlled Gaussian noise and SSIM-based loss. Outputs are sequentially numbered images
and labels, ready for YOLO training.
"""

# ------------------------------
# Install & Imports
# ------------------------------
!pip install -q tensorflow scikit-image

import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

# ------------------------------
# CONFIGURATION
# ------------------------------
IMG_SIZE = 128
NOISE_STD_DEV = 0.07
SSIM_WEIGHT = 0.75
EPOCHS = 30
BATCH_SIZE = 16
NUM_DIGITS = 3  # For sequential filenames (e.g., color_001.png)

BASE_DIR = "/content/tiget"  # Original images + labels
OUTPUT_DIR = "/content/gdrive/MyDrive/cae_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Data Loading
# ------------------------------
def load_images_from_folder(folder):
    imgs, filenames = [], []
    if not os.path.exists(folder):
        raise RuntimeError(f"Source folder not found: {folder}")
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".png"):
            img = cv2.imread(os.path.join(folder, fname))
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            imgs.append(img)
            filenames.append(os.path.splitext(fname)[0])
    return np.array(imgs), filenames

print("Loading images...")
all_imgs, all_names = load_images_from_folder(BASE_DIR)

if len(all_imgs) == 0:
    raise RuntimeError("No images loaded. Check your dataset path!")

train_imgs, val_imgs, train_names, val_names = train_test_split(
    all_imgs, all_names, test_size=0.2, random_state=42
)
print(f"Total images loaded: {len(all_imgs)}")

# ------------------------------
# CAE Model
# ------------------------------
def build_sharp_cae():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2,2))(c3)

    # Bottleneck
    b = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    noisy_b = layers.GaussianNoise(NOISE_STD_DEV)(b)

    # Decoder with weakened skip connections
    f_c3 = layers.Conv2D(128, (1,1), activation='relu', padding='same')(c3)
    u3 = layers.UpSampling2D((2,2))(noisy_b)
    m3 = layers.Concatenate()([u3, f_c3])
    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(m3)

    f_c2 = layers.Conv2D(64, (1,1), activation='relu', padding='same')(c2)
    u2 = layers.UpSampling2D((2,2))(c4)
    m2 = layers.Concatenate()([u2, f_c2])
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(m2)

    f_c1 = layers.Conv2D(32, (1,1), activation='relu', padding='same')(c1)
    u1 = layers.UpSampling2D((2,2))(c5)
    m1 = layers.Concatenate()([u1, f_c1])
    c6 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(m1)

    out = layers.Conv2D(3, (1,1), activation='sigmoid', padding='same')(c6)
    return models.Model(inp, out)

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return (1.0 - SSIM_WEIGHT) * mse + SSIM_WEIGHT * ssim_loss

# ------------------------------
# Train CAE
# ------------------------------
sharp_cae = build_sharp_cae()
sharp_cae.compile(optimizer='adam', loss=combined_loss)
print("Starting CAE training...")
sharp_cae.fit(train_imgs, train_imgs, validation_data=(val_imgs, val_imgs),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# ------------------------------
# Generate & Save Augmented Images
# ------------------------------
print("Generating augmented images...")

start_idx = len(all_imgs) + 1
current_idx = start_idx

out_images_dir = os.path.join(OUTPUT_DIR, "images")
out_labels_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_labels_dir, exist_ok=True)

for imgs, names in [(train_imgs, train_names), (val_imgs, val_names)]:
    preds = sharp_cae.predict(imgs)
    for i, img in enumerate(preds):
        new_base = f"color_{current_idx:0{NUM_DIGITS}d}"
        # Save image
        cv2.imwrite(os.path.join(out_images_dir, f"{new_base}.png"), (img*255).astype(np.uint8))
        # Copy label
        orig_label = os.path.join(BASE_DIR, f"{names[i]}.txt")
        new_label = os.path.join(out_labels_dir, f"{new_base}.txt")
        if os.path.exists(orig_label):
            shutil.copy(orig_label, new_label)
        else:
            print(f"⚠️ Label missing for {names[i]}")
        current_idx += 1

print(f"✅ Augmented images saved to: {OUTPUT_DIR}")

# ------------------------------
# Sample SSIM Check
# ------------------------------
avg_ssim, count = 0, 0
sample_size = min(len(train_imgs), 50)
sample_idx = np.random.choice(len(train_imgs), sample_size, replace=False)

for i in sample_idx:
    orig = (train_imgs[i]*255).astype(np.uint8)
    pred_img = (sharp_cae.predict(np.expand_dims(train_imgs[i],0))[0]*255).astype(np.uint8)
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(orig_gray, pred_gray, full=True)
    avg_ssim += score
    count += 1

final_avg_ssim = avg_ssim / count
print(f"\n✨ Average SSIM (sampled): {final_avg_ssim:.4f}")
if final_avg_ssim < 0.98:
    print("✅ SSIM indicates good augmentation effect for YOLO training.")
else:
    print("⚠️ SSIM too high: augmentation effect may be subtle.")

# ------------------------------
# Visualize 3 Images Side by Side
# ------------------------------
num_display = min(3, len(train_imgs))
plt.figure(figsize=(10, 8))
for i in range(num_display):
    plt.subplot(2, num_display, i+1)
    plt.imshow((train_imgs[i]*255).astype(np.uint8))
    plt.axis('off'); plt.title("Original")
    
    aug_img_display = sharp_cae.predict(np.expand_dims(train_imgs[i],0))[0]
    plt.subplot(2, num_display, i+1+num_display)
    plt.imshow((aug_img_display*255).astype(np.uint8))
    plt.axis('off'); plt.title("Augmented")

plt.tight_layout()
plt.show()
