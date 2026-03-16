# Construction Site Segmentation (U-Net)

A deep learning pipeline for **semantic segmentation of construction sites from aerial images** using a **U-Net architecture implemented in TensorFlow**.

The project includes:

* Dataset preprocessing from ZIP files
* Image–mask validation
* Data augmentation
* Automatic dataset splitting
* U-Net training pipeline
* Model evaluation
* Saving trained model and logs

This script is designed to run on a **local machine, research server, or Docker container**.

---

# Project Workflow

The training pipeline performs the following steps:

1. Load dataset ZIP files
2. Extract images and masks
3. Match image–mask pairs
4. Validate dataset
5. Normalize images and masks
6. Resize images to **512 × 512**
7. Apply data augmentation
8. Split dataset into:

   * Train
   * Validation
   * Test
9. Train a **U-Net segmentation model**
10. Evaluate model performance
11. Save model and training history

---

# Project Structure

Example repository structure:

```
construction-site-segmentation/
│
├── server_notebook.py
├── README.md
├── requirements.txt
│
├── data/
│   └── input/
│       ├── original_images.zip
│       └── mask_images.zip
│
└── segmentation_results/
    ├── train/
    ├── val/
    ├── test/
    ├── metadata/
    ├── logs/
    └── models/
```

---

# Dataset Format

The project expects **two ZIP files**:

### Original Images

```
original_images.zip
```

Contains aerial RGB images.

### Mask Images

```
mask_images.zip
```

Contains segmentation masks.

Important requirement:

Each mask must have **the same filename as its corresponding image**.

Example:

```
image001.png
image001.png (mask)
```

Supported formats:

* `.png`
* `.jpg`
* `.jpeg`

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/construction-site-segmentation.git
cd construction-site-segmentation
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Required Libraries

Main dependencies:

* TensorFlow
* NumPy
* OpenCV
* Albumentations
* Scikit-learn
* tqdm
* Pillow

Example `requirements.txt`:

```
tensorflow
numpy
opencv-python
albumentations
scikit-learn
tqdm
pillow
```

---

# Running the Training Pipeline

Place dataset ZIP files in:

```
data/input/
```

Then run:

```
python server_notebook.py
```

---

# Data Preprocessing

The script automatically performs:

### Dataset Extraction

ZIP files are extracted into temporary directories.

### Pair Matching

Images and masks are matched using their filenames.

### Image Normalization

Images are scaled to:

```
[0,1]
```

### Mask Normalization

Masks are converted to **binary values**:

```
0 → background
1 → construction site
```

### Resizing

All images are resized to:

```
512 × 512
```

### Data Augmentation

The following augmentations are applied:

* Horizontal Flip
* Vertical Flip
* Rotation
* Perspective Transform
* Gaussian Noise
* Color Jitter
* Random Rain
* Random Fog
* Gaussian Blur
* Coarse Dropout

---

# Dataset Splitting

Default dataset split:

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 75%        |
| Validation | 15%        |
| Test       | 10%        |

---

# Model Architecture

The project uses a **U-Net convolutional neural network** for semantic segmentation.

Architecture components:

Encoder

* Convolution blocks
* Batch normalization
* Max pooling

Bottleneck

* Deep convolution layers

Decoder

* Transposed convolution
* Skip connections

Output layer

* Sigmoid activation

Input shape:

```
512 × 512 × 3
```

---

# Training Configuration

Default training parameters:

| Parameter     | Value               |
| ------------- | ------------------- |
| Epochs        | 50                  |
| Batch Size    | 32                  |
| Optimizer     | Adam                |
| Learning Rate | 0.001               |
| Loss Function | Binary Crossentropy |

Metrics used:

* Accuracy
* Intersection over Union (IoU)
* Precision
* Recall

Training also uses:

* EarlyStopping
* ReduceLROnPlateau

---

# Output

After training, results are saved in:

```
segmentation_results/
```

Directory contents:

### Processed Dataset

```
train/
val/
test/
```

### Model

```
models/construction_site_segmentation.h5
```

### Training Logs

```
logs/training_history.json
```

### Dataset Metadata

```
metadata/dataset_info.json
```

---

# Example Training Output

```
Loss:      0.2453
Accuracy:  0.9124
IoU:       0.7321
Precision: 0.8845
Recall:    0.8612
```

---

# Using Docker (Optional)

Build the Docker image:

```
docker build -t segmentation-training .
```

Run container:

```
docker run \
-v /server/data:/app/data/input \
-v /server/output:/app/segmentation_results \
segmentation-training
```

---

# Next Steps

After training:

1. Download the `segmentation_results` folder
2. Use the trained model for inference
3. Analyze training curves in `training_history.json`
4. Visualize segmentation predictions

Possible improvements:

* Attention U-Net
* Multi-class segmentation
* Mixed precision training
* GPU optimization
* Deployment API

---

# License

This project is released under the **MIT License**.

---

# Author

Developed for **Construction Site Segmentation using Deep Learning and U-Net**.
