"""
GOOGLE COLAB NOTEBOOK - Ready to Copy & Paste
Construction Site Segmentation with Google Drive

Instructions:
1. Go to colab.research.google.com
2. Create new notebook
3. Copy each section below into separate cells
4. Run cells in order
5. Follow prompts (Drive authentication, etc.)
"""

# ============================================================================
# CELL 1: Mount Google Drive
# ============================================================================
"""
In Colab: Copy this entire cell and run it
"""

from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')
print("✓ Google Drive mounted successfully!")

# ============================================================================
# CELL 2: Configure Paths and Verify Data
# ============================================================================
"""
Update DRIVE_BASE if your folder structure is different
"""

# UPDATE THIS TO YOUR FOLDER NAME
DRIVE_BASE = '/content/drive/My Drive/MyProject'

# Data paths
ORIGINAL_IMAGES_DIR = f'{DRIVE_BASE}/data/original'
MASK_IMAGES_DIR = f'{DRIVE_BASE}/data/masks'
PROCESSED_DATA_DIR = f'{DRIVE_BASE}/data/processed'
MODELS_DIR = f'{DRIVE_BASE}/models'
MODEL_SAVE_PATH = f'{MODELS_DIR}/construction_site_segmentation.h5'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Verify data exists
print("\n" + "="*60)
print("VERIFYING DATA")
print("="*60)

if os.path.exists(ORIGINAL_IMAGES_DIR):
    orig_count = len(os.listdir(ORIGINAL_IMAGES_DIR))
    print(f"✓ Original images: {orig_count}")
else:
    print(f"✗ Original images directory not found: {ORIGINAL_IMAGES_DIR}")
    print("  Create this folder in Drive and upload your images")

if os.path.exists(MASK_IMAGES_DIR):
    mask_count = len(os.listdir(MASK_IMAGES_DIR))
    print(f"✓ Mask images: {mask_count}")
else:
    print(f"✗ Mask images directory not found: {MASK_IMAGES_DIR}")
    print("  Create this folder in Drive and upload your masks")

print(f"\nPaths configured:")
print(f"  Original: {ORIGINAL_IMAGES_DIR}")
print(f"  Masks: {MASK_IMAGES_DIR}")
print(f"  Processed: {PROCESSED_DATA_DIR}")
print(f"  Model: {MODEL_SAVE_PATH}")

# ============================================================================
# CELL 3: Install Dependencies
# ============================================================================
"""
This may take 2-3 minutes
"""

print("Installing dependencies...")
!pip install -q tensorflow keras numpy opencv-python albumentations scikit-learn matplotlib tqdm

print("✓ All dependencies installed!")
print("\nTensorFlow version:", __import__('tensorflow').__version__)

# ============================================================================
# CELL 4: Define DatasetPreprocessor Class
# ============================================================================
"""
IMPORTANT: Copy the ENTIRE content of preprocess_pipeline.py here
This is a simplified version - copy the full file for best results
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm

class DatasetPreprocessor:
    """Comprehensive preprocessing pipeline for construction site dataset."""
    
    def __init__(self, original_images_dir, mask_images_dir, output_dir, 
                 target_size=(512, 512), seed=42):
        self.original_dir = Path(original_images_dir)
        self.mask_dir = Path(mask_images_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.seed = seed
        np.random.seed(seed)
        
        self.create_output_structure()
        self.stats = {
            'total_images': 0,
            'valid_pairs': 0,
            'invalid_pairs': [],
            'preprocessing_log': []
        }
    
    def create_output_structure(self):
        """Create organized directory structure."""
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    
    def find_image_pairs(self):
        """Find matching original-mask pairs."""
        original_images = set(f.stem for f in self.original_dir.glob('*') 
                            if f.suffix.lower() in ['.png', '.jpg', '.jpeg'])
        mask_images = set(f.stem for f in self.mask_dir.glob('*') 
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg'])
        
        matched_pairs = original_images & mask_images
        self.stats['total_images'] = len(matched_pairs)
        
        print(f"✓ Found {len(matched_pairs)} matching image-mask pairs")
        return list(matched_pairs)
    
    def load_and_validate_pair(self, image_stem):
        """Load and validate image-mask pair."""
        original_file = list(self.original_dir.glob(f'{image_stem}.*'))
        mask_file = list(self.mask_dir.glob(f'{image_stem}.*'))
        
        if not original_file or not mask_file:
            self.stats['invalid_pairs'].append(image_stem)
            return None, None, False
        
        try:
            image = cv2.imread(str(original_file[0]))
            mask = cv2.imread(str(mask_file[0]), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                return None, None, False
            
            if image.shape[:2] != mask.shape:
                return None, None, False
            
            self.stats['valid_pairs'] += 1
            return image, mask, True
        except:
            return None, None, False
    
    def normalize_image(self, image):
        """Normalize image to [0, 1]."""
        return image.astype(np.float32) / 255.0
    
    def normalize_mask(self, mask):
        """Normalize mask to binary."""
        _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return mask_binary.astype(np.float32)
    
    def resize_image_and_mask(self, image, mask):
        """Resize to target size."""
        h, w = self.target_size
        image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized
    
    def create_augmentation_pipeline(self, augment=True):
        """Create augmentation pipeline."""
        if not augment:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            ], is_check_shapes=False)
        
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.GaussNoise(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.GaussBlur(blur_limit=3, p=0.2),
            A.Cutout(num_holes=4, max_h_size=30, max_w_size=30, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ], is_check_shapes=False)
    
    def process_dataset(self, augment_training=True, val_size=0.15, test_size=0.1):
        """Main preprocessing pipeline."""
        print("\n" + "="*60)
        print("PREPROCESSING DATASET")
        print("="*60)
        
        # Find pairs
        pairs = self.find_image_pairs()
        
        # Split
        train_pairs, temp = train_test_split(
            pairs, test_size=(val_size + test_size), random_state=self.seed
        )
        val_pairs, test_pairs = train_test_split(
            temp, test_size=(test_size / (val_size + test_size)), random_state=self.seed
        )
        
        print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
        
        # Process splits
        split_data = {
            'train': (train_pairs, True),
            'val': (val_pairs, False),
            'test': (test_pairs, False)
        }
        
        all_metadata = {'train': {}, 'val': {}, 'test': {}}
        
        for split_name, (split_pairs, do_augment) in split_data.items():
            augmentation = self.create_augmentation_pipeline(
                augment=(do_augment and augment_training)
            )
            
            processed_count = 0
            for image_stem in tqdm(split_pairs, desc=f"Processing {split_name}"):
                image, mask, valid = self.load_and_validate_pair(image_stem)
                
                if not valid:
                    continue
                
                image = self.normalize_image(image)
                mask = self.normalize_mask(mask)
                image, mask = self.resize_image_and_mask(image, mask)
                
                augmented = augmentation(image=image, mask=mask)
                image_aug = augmented['image']
                mask_aug = augmented['mask']
                
                split_dir = self.output_dir / split_name
                image_path = split_dir / 'images' / f'{image_stem}.npy'
                mask_path = split_dir / 'masks' / f'{image_stem}.npy'
                
                np.save(image_path, image_aug.astype(np.float32))
                np.save(mask_path, mask_aug.astype(np.float32))
                
                all_metadata[split_name][image_stem] = {
                    'original_shape': image.shape,
                    'processed_shape': image_aug.shape,
                    'mask_pixels': float(np.sum(mask_aug)),
                    'mask_percentage': float(np.sum(mask_aug) / mask_aug.size * 100)
                }
                
                processed_count += 1
            
            print(f"  ✓ {processed_count} images processed for {split_name}")
        
        # Save metadata
        metadata_file = self.output_dir / 'metadata' / 'dataset_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Valid pairs: {self.stats['valid_pairs']}")
        print(f"  Output: {self.output_dir}")
        
        return all_metadata

class DataLoader:
    """Load preprocessed dataset."""
    
    def __init__(self, dataset_dir, batch_size=32):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
    
    def create_tf_dataset(self, split='train', shuffle=True):
        """Create TensorFlow dataset."""
        import tensorflow as tf
        
        split_dir = self.dataset_dir / split
        image_files = sorted((split_dir / 'images').glob('*.npy'))
        
        def load_image_mask(image_path):
            image = np.load(image_path.numpy().decode('utf-8'))
            mask_path = image_path.numpy().decode('utf-8').replace('/images/', '/masks/')
            mask = np.load(mask_path)
            return image, mask
        
        dataset = tf.data.Dataset.from_tensor_slices([str(f) for f in image_files])
        dataset = dataset.map(
            lambda x: tf.py_function(load_image_mask, [x], [tf.float32, tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(image_files)))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

print("✓ DatasetPreprocessor and DataLoader classes loaded!")

# ============================================================================
# CELL 5: Define U-Net Model Class
# ============================================================================
"""
IMPORTANT: Copy the ENTIRE content of keras_model.py here
This is a simplified version - copy the full file for best results
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class UNetSegmentation(models.Model):
    """U-Net architecture for semantic segmentation."""
    
    def __init__(self, input_shape=(512, 512, 3), num_classes=1, filters_base=32):
        super().__init__()
        self.input_shape_val = input_shape
        self.num_classes = num_classes
        self.filters_base = filters_base
        
        # Encoder
        self.conv1 = self.conv_block(filters_base, name='conv1')
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv2 = self.conv_block(filters_base * 2, name='conv2')
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv3 = self.conv_block(filters_base * 4, name='conv3')
        self.pool3 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv4 = self.conv_block(filters_base * 8, name='conv4')
        self.pool4 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        # Bottleneck
        self.conv5 = self.conv_block(filters_base * 16, name='conv5')
        
        # Decoder
        self.upconv4 = layers.Conv2DTranspose(filters_base * 8, 3, strides=2, padding='same', activation='relu')
        self.conv6 = self.conv_block(filters_base * 8, name='conv6')
        
        self.upconv3 = layers.Conv2DTranspose(filters_base * 4, 3, strides=2, padding='same', activation='relu')
        self.conv7 = self.conv_block(filters_base * 4, name='conv7')
        
        self.upconv2 = layers.Conv2DTranspose(filters_base * 2, 3, strides=2, padding='same', activation='relu')
        self.conv8 = self.conv_block(filters_base * 2, name='conv8')
        
        self.upconv1 = layers.Conv2DTranspose(filters_base, 3, strides=2, padding='same', activation='relu')
        self.conv9 = self.conv_block(filters_base, name='conv9')
        
        # Output
        self.output_conv = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')
    
    def conv_block(self, filters, name):
        """Convolutional block."""
        return models.Sequential([
            layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_conv1'),
            layers.BatchNormalization(name=f'{name}_bn1'),
            layers.Dropout(0.2),
            layers.Conv2D(filters, 3, padding='same', activation='relu', name=f'{name}_conv2'),
            layers.BatchNormalization(name=f'{name}_bn2'),
        ], name=name)
    
    def call(self, inputs, training=False):
        """Forward pass."""
        c1 = self.conv1(inputs, training=training)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1, training=training)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2, training=training)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3, training=training)
        p4 = self.pool4(c4)
        
        c5 = self.conv5(p4, training=training)
        
        up4 = self.upconv4(c5)
        up4 = layers.Concatenate()([up4, c4])
        c6 = self.conv6(up4, training=training)
        
        up3 = self.upconv3(c6)
        up3 = layers.Concatenate()([up3, c3])
        c7 = self.conv7(up3, training=training)
        
        up2 = self.upconv2(c7)
        up2 = layers.Concatenate()([up2, c2])
        c8 = self.conv8(up2, training=training)
        
        up1 = self.upconv1(c8)
        up1 = layers.Concatenate()([up1, c1])
        c9 = self.conv9(up1, training=training)
        
        output = self.output_conv(c9)
        return output

class SegmentationModel:
    """High-level segmentation model API."""
    
    def __init__(self, input_shape=(512, 512, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build(self, filters_base=32):
        """Build model."""
        self.model = UNetSegmentation(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            filters_base=filters_base
        )
        self.model.build(input_shape=(None, *self.input_shape))
        print("✓ Model built successfully")
    
    def compile(self, learning_rate=1e-3):
        """Compile model."""
        loss = keras.losses.BinaryCrossentropy()
        metrics = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.MeanIoU(num_classes=2, name='iou'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("✓ Model compiled")
    
    def train(self, train_dataset, val_dataset, epochs=50, callbacks=None):
        """Train model."""
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
                ),
            ]
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return self.history
    
    def predict_mask(self, image):
        """Predict mask."""
        image_batch = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image_batch, verbose=0)
        mask = np.squeeze(prediction, axis=0)
        mask = (mask > 0.5).astype(np.uint8) * 255
        return mask
    
    def save(self, filepath):
        """Save model."""
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model."""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")
    
    def evaluate(self, test_dataset):
        """Evaluate model."""
        results = self.model.evaluate(test_dataset, verbose=0)
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Loss:      {results[0]:.4f}")
        print(f"Accuracy:  {results[1]:.4f}")
        print(f"IoU:       {results[2]:.4f}")
        print(f"Precision: {results[3]:.4f}")
        print(f"Recall:    {results[4]:.4f}")
        print("="*60)
        return results

print("✓ U-Net model classes loaded!")

# ============================================================================
# CELL 6: Preprocess Dataset
# ============================================================================
"""
This will take 30 minutes to 1 hour depending on image sizes
"""

print("\n" + "="*70)
print("PREPROCESSING DATASET FROM GOOGLE DRIVE")
print("="*70)

preprocessor = DatasetPreprocessor(
    original_images_dir=ORIGINAL_IMAGES_DIR,
    mask_images_dir=MASK_IMAGES_DIR,
    output_dir=PROCESSED_DATA_DIR,
    target_size=(512, 512),
    seed=42
)

metadata = preprocessor.process_dataset(
    augment_training=True,
    val_size=0.15,
    test_size=0.10
)

print("\n✓ Preprocessing complete!")
print("  Processed data saved to Google Drive")

# ============================================================================
# CELL 7: Load Data for Training
# ============================================================================
"""
This loads the preprocessed data into TensorFlow datasets
"""

print("\nLoading datasets for training...")

data_loader = DataLoader(dataset_dir=PROCESSED_DATA_DIR, batch_size=32)

train_dataset = data_loader.create_tf_dataset(split='train', shuffle=True)
val_dataset = data_loader.create_tf_dataset(split='val', shuffle=False)
test_dataset = data_loader.create_tf_dataset(split='test', shuffle=False)

print("✓ Datasets loaded successfully!")

# ============================================================================
# CELL 8: Build and Compile Model
# ============================================================================
"""
Creates the U-Net model architecture
"""

print("\n" + "="*70)
print("BUILDING MODEL")
print("="*70)

model = SegmentationModel(input_shape=(512, 512, 3), num_classes=1)
model.build(filters_base=32)
model.compile(learning_rate=0.001)

# ============================================================================
# CELL 9: Train Model
# ============================================================================
"""
Training will take 2-3 hours on GPU
Monitor the metrics - they should improve steadily
"""

print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)
print("This will take 2-3 hours on GPU...")
print()

history = model.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50
)

print("\n✓ Training complete!")

# ============================================================================
# CELL 10: Evaluate on Test Set
# ============================================================================
"""
Evaluate model performance on unseen test data
"""

print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

model.evaluate(test_dataset)

# ============================================================================
# CELL 11: Save Model to Google Drive
# ============================================================================
"""
Model is automatically saved to Google Drive
"""

print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model.save(MODEL_SAVE_PATH)

print(f"\n✓ SUCCESS!")
print(f"  Model saved to: {MODEL_SAVE_PATH}")
print(f"\n  You can now:")
print(f"  1. Download the model from Google Drive")
print(f"  2. Use it for inference on new images")
print(f"  3. Share it with collaborators")

# ============================================================================
# CELL 12 (Optional): Plot Training History
# ============================================================================
"""
Visualize how the model improved during training
"""

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train')
axes[0, 1].plot(history.history['val_accuracy'], label='Val')
axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# IoU
axes[1, 0].plot(history.history['iou'], label='Train')
axes[1, 0].plot(history.history['val_iou'], label='Val')
axes[1, 0].set_title('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Precision & Recall
axes[1, 1].plot(history.history['precision'], label='Precision')
axes[1, 1].plot(history.history['recall'], label='Recall')
axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{MODELS_DIR}/training_history.png', dpi=150, bbox_inches='tight')
print(f"✓ Training history plot saved to {MODELS_DIR}/training_history.png")
plt.show()

# ============================================================================
# CELL 13 (Optional): Make Predictions on Test Images
# ============================================================================
"""
Use the trained model to make predictions on test images
"""

# Make predictions on first test image
print("\nMaking predictions on test set...")

# Get a batch from test dataset
for images, masks in test_dataset.take(1):
    # Make predictions
    predictions = model.model.predict(images[:4], verbose=0)
    
    # Visualize
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    
    for i in range(4):
        # Original image
        axes[i, 0].imshow((images[i].numpy() * 255).astype(np.uint8))
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title(f'Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{MODELS_DIR}/predictions_sample.png', dpi=100, bbox_inches='tight')
    print(f"✓ Sample predictions saved")
    plt.show()

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
print(f"\nYour trained model is saved in Google Drive:")
print(f"  {MODEL_SAVE_PATH}")
print(f"\nNext steps:")
print(f"  1. Download the model to your computer")
print(f"  2. Use it for inference on new images")
print(f"  3. Fine-tune with more data if needed")
