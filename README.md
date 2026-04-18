# Multi-Class Image Segmentation using U-Net (Drone Dataset)

## Overview

This project implements a **multi-class semantic segmentation pipeline** using a U-Net architecture in TensorFlow.
It is designed for drone imagery and supports segmentation into multiple classes such as:

- Background
- Water
- Vegetation
- Structure (buildings)

The pipeline includes:

- Data preprocessing (RGB masks → label masks)
- Augmentation using Albumentations
- Custom Dice + CrossEntropy loss
- Mean IoU evaluation metric
- Training visualization (loss, accuracy, IoU)
- Prediction on new images using a separate script

---
## Dataset Download Link

https://drive.google.com/uc?id=1b88NGOW-7EgNQ1LLI0UHXE-KLOzaqnGm

## Project Structure

```id="gi40e9"
├── main.py              # Training pipeline
├── predict.py           # Inference script
├── data.zip             # Dataset (images + masks)
├── processed/           # Preprocessed dataset (auto-generated)
├── predictions/         # Output predictions
├── final_model.h5       # Saved trained model
└── README.md
```

---

## GPU Setup (CUDA + cuDNN)

To use GPU acceleration, you need to install:

- NVIDIA GPU drivers
- CUDA Toolkit
- cuDNN (CUDA Deep Neural Network library)

### 1. Check GPU

Run:

```id="xt7i92"
nvidia-smi
```

If your GPU is listed, proceed.

---

### 2. Install CUDA

Download CUDA Toolkit from:
https://developer.nvidia.com/cuda-downloads

Install a version compatible with your TensorFlow version.

---

### 3. Install cuDNN

Download cuDNN from:
https://developer.nvidia.com/cudnn

Steps:

1. Extract the cuDNN folder
2. Copy contents into CUDA directory:

```id="298eti"
bin → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin
lib → ...\lib
include → ...\include
```

---

### 4. Set Environment Variables

Add CUDA paths to system environment variables:

```id="q8bh7o"
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\libnvvp
```

---

### 5. Verify TensorFlow GPU

Run Python:

```python id="1f8cyq"
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If GPU appears → setup is correct.

---

## Virtual Environment Setup

Create and activate a virtual environment:

### Windows

```id="v5odfj"
python -m venv .venv
.venv\Scripts\activate
```

### Linux / Mac

```id="ooopzd"
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Requirements

Install dependencies:

```id="da8lb2"
pip install -r requirements.txt
```

If you don’t have a requirements file, install manually:

```id="yaiq7u"
pip install tensorflow opencv-python numpy matplotlib albumentations scikit-learn seaborn
```

---

## Dataset Format

Your dataset should be structured inside `data.zip`:

```id="plalnn"
input/
   ├── original_images/
   │      img1.jpg
   │      img2.jpg
   ├── masked_images/
          img1.jpg
          img2.jpg
```

- Each mask is a **color-coded segmentation image**
- Colors are mapped to class labels during preprocessing

---

## Training the Model

Run:

```id="vjw66v"
python main.py
```

This will:

- Extract dataset
- Preprocess images and masks
- Train the U-Net model
- Save model as `final_model.h5`
- Display training graphs (loss, accuracy, IoU)

---

## Metrics

The model reports:

- Accuracy (pixel-wise)
- Mean IoU (Intersection over Union)

---

## Prediction (predict.py)

This script performs inference on new images.

### Input Options

You can use:

- A single image
- A folder of images

Modify in `predict.py`:

```id="rbqbeh"
INPUT_PATH = "test_images"
```

---

### Run Prediction

```id="p0ve8p"
python predict.py
```

---

### Output

Predictions are saved in:

```id="w1dlt2"
predictions/
   image1_mask.png
   image2_mask.png
```

Each output includes:

- Segmented mask (colored)
- Optional visualization (original + prediction)

---

## Class Mapping

| Label | Class      | Color (RGB)   |
| ----- | ---------- | ------------- |
| 0     | Background | (169,169,169) |
| 1     | Water      | (14,135,204)  |
| 2     | Vegetation | (124,252,0)   |
| 3     | Structure  | (155,38,182)  |

---

## Future Improvements

- Replace U-Net with DeepLabV3+
- Add class-wise weighting
- Real-time segmentation (video/webcam)
- Better class separation (e.g., road vs background)
