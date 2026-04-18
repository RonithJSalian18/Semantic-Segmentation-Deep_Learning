"""
PREDICTION SCRIPT FOR SEGMENTATION
✔ Works with trained model
✔ Supports single image & folder
✔ Saves output masks
✔ Visualizes results
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIG =================
MODEL_PATH = (r"D:\Semantic\models\final_model(82).h5")
IMG_SIZE = (256, 256)

INPUT_PATH = (r"D:\Downloads\dl\classes_dataset\input\original_images\060.png")   # folder OR single image
OUTPUT_DIR = "predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= CLASS COLORS =================
CLASS_COLORS = np.array([
    [169,169,169],   # background
    [14,135,204],    # water
    [124,252,0],     # vegetation
    [155,38,182],    # structure
], dtype=np.uint8)

# ================= LOAD MODEL =================
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# ================= PREPROCESS =================
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()

    img = cv2.resize(img, IMG_SIZE)
    img = (img / 255.0 - 0.5) * 2

    return img.astype(np.float32), original

# ================= PREDICT =================
def predict_image(img_path, save=True, show=True):
    img, original = preprocess(img_path)

    pred = model.predict(np.expand_dims(img, 0))[0]
    pred = np.argmax(pred, axis=-1)

    color_mask = CLASS_COLORS[pred]

    filename = Path(img_path).stem

    if save:
        cv2.imwrite(
            f"{OUTPUT_DIR}/{filename}_mask.png",
            cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        )

    if show:
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(pred)
        plt.title("Label Mask")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(color_mask)
        plt.title("Colored Mask")
        plt.axis("off")

        plt.show()

# ================= RUN =================
def run():
    if os.path.isdir(INPUT_PATH):
        images = list(Path(INPUT_PATH).glob("*"))
        print(f"📂 Found {len(images)} images")

        for img_path in images:
            print(f"🔍 Processing: {img_path}")
            predict_image(str(img_path), save=True, show=False)

        print("✅ All predictions saved")

    else:
        print(f"🔍 Processing single image: {INPUT_PATH}")
        predict_image(INPUT_PATH, save=True, show=True)

# ================= MAIN =================
if __name__ == "__main__":
    run()