"""
SERVER TRAINING NOTEBOOK - ZIP File Input, Local Output
Construction Site Segmentation

This is the Colab notebook modified for college server deployment.
Instead of reading from Google Drive, it reads ZIP files uploaded to the server.

Instructions:
1. Upload original_images.zip to server
2. Upload mask_images.zip to server
3. Run this script with: python server_notebook.py
   OR copy cells to Jupyter notebook and run sequentially
"""

# ============================================================================
# CELL 1: Import Libraries and Setup Paths
# ============================================================================

import os
import sys
import cv2
import numpy as np
import json
import zipfile
import tempfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully!")

# ============================================================================
# CELL 2: Configure ZIP File Paths and Output Directory
# ============================================================================

"""
CONFIGURE THESE PATHS BASED ON YOUR SERVER SETUP
"""

# Input ZIP files (upload these to your server)
ORIGINAL_IMAGES_ZIP = "./data/input/original_images.zip"  # Your aerial images ZIP
MASKS_ZIP = "./data/input/mask_images.zip"                 # Your masks ZIP

# Output directory (all results will be saved here)
OUTPUT_DIR = "./segmentation_results"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("CONFIGURATION")
print("="*70)
print(f"Input ZIP files:")
print(f"  Original: {os.path.abspath(ORIGINAL_IMAGES_ZIP)}")
print(f"  Masks:    {os.path.abspath(MASKS_ZIP)}")
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

# Check if ZIP files exist
if not os.path.exists(ORIGINAL_IMAGES_ZIP):
    print(f"\n✗ ERROR: {ORIGINAL_IMAGES_ZIP} not found!")
    print("  Please upload original_images.zip to ./data/input/")
    sys.exit(1)

if not os.path.exists(MASKS_ZIP):
    print(f"\n✗ ERROR: {MASKS_ZIP} not found!")
    print("  Please upload mask_images.zip to ./data/input/")
    sys.exit(1)

print(f"\n✓ ZIP files found!")

# ============================================================================
# CELL 3: Define ServerDatasetPreprocessor Class
# ============================================================================

class ServerDatasetPreprocessor:
    """Preprocessing pipeline that reads from ZIP files."""
    
    def __init__(self, original_zip, masks_zip, output_dir, target_size=(512, 512), seed=42):
        self.original_zip = Path(original_zip)
        self.masks_zip = Path(masks_zip)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.seed = seed
        np.random.seed(seed)
        
        # Temporary directories for extraction
        self.temp_original = None
        self.temp_masks = None
        
        # Create output structure
        self.create_output_structure()
        
        self.stats = {
            'total_images': 0,
            'valid_pairs': 0,
            'invalid_pairs': [],
        }
    
    def create_output_structure(self):
        """Create organized output directory structure."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        print(f"✓ Output structure created: {self.output_dir}")
    
    def extract_zips(self):
        """Extract ZIP files to temporary directories."""
        print("\n" + "="*60)
        print("EXTRACTING ZIP FILES")
        print("="*60)
        
        # Create temporary directories
        self.temp_original = Path(tempfile.mkdtemp(prefix="original_"))
        self.temp_masks = Path(tempfile.mkdtemp(prefix="masks_"))
        
        print(f"Extracting original images...")
        with zipfile.ZipFile(self.original_zip, 'r') as zip_ref:
            zip_ref.extractall(self.temp_original)
        print(f"✓ Extracted to {self.temp_original}")
        
        print(f"\nExtracting mask images...")
        with zipfile.ZipFile(self.masks_zip, 'r') as zip_ref:
            zip_ref.extractall(self.temp_masks)
        print(f"✓ Extracted to {self.temp_masks}")
    
    def find_image_pairs(self):
        """Find matching original-mask pairs from extracted directories."""
        print("\n" + "="*60)
        print("FINDING IMAGE-MASK PAIRS")
        print("="*60)
        
        original_images = {}
        mask_images = {}
        
        # Find all image files recursively
        for file in self.temp_original.rglob('*'):
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg'] and file.is_file():
                original_images[file.stem] = file
        
        for file in self.temp_masks.rglob('*'):
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg'] and file.is_file():
                mask_images[file.stem] = file
        
        # Find matching pairs
        matched_pairs = {}
        for stem in original_images.keys():
            if stem in mask_images:
                matched_pairs[stem] = (original_images[stem], mask_images[stem])
        
        self.stats['total_images'] = len(matched_pairs)
        
        print(f"✓ Original images found: {len(original_images)}")
        print(f"✓ Mask images found: {len(mask_images)}")
        print(f"✓ Matched pairs: {len(matched_pairs)}")
        
        return matched_pairs
    
    def load_and_validate_pair(self, stem, original_path, mask_path):
        """Load and validate image-mask pair."""
        try:
            image = cv2.imread(str(original_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                self.stats['invalid_pairs'].append(f"{stem}: Could not read")
                return None, None, False
            
            if image.shape[:2] != mask.shape:
                self.stats['invalid_pairs'].append(f"{stem}: Shape mismatch")
                return None, None, False
            
            self.stats['valid_pairs'] += 1
            return image, mask, True
            
        except Exception as e:
            self.stats['invalid_pairs'].append(f"{stem}: {str(e)}")
            return None, None, False
    
    def normalize_image(self, image):
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def normalize_mask(self, mask):
        """Normalize mask to binary [0, 1]."""
        _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return mask_binary.astype(np.float32)
    
    def resize_image_and_mask(self, image, mask):
        """Resize to target size."""
        h, w = self.target_size
        image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized
    
    def create_augmentation_pipeline(self, augment=True):
        """Create data augmentation pipeline."""
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
            # FIXED: Using CoarseDropout instead of Cutout
            A.CoarseDropout(max_holes=4, max_height=30, max_width=30, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ], is_check_shapes=False)
    
    def process_dataset(self, augment_training=True, val_size=0.15, test_size=0.1):
        """Main preprocessing pipeline."""
        print("\n" + "="*60)
        print("PREPROCESSING DATASET")
        print("="*60)
        
        # Extract ZIPs
        self.extract_zips()
        
        # Find pairs
        print("\n[Step 1/4] Finding matching pairs...")
        pairs = self.find_image_pairs()
        pair_list = list(pairs.keys())
        
        # Split dataset
        print("\n[Step 2/4] Splitting dataset...")
        train_pairs, temp = train_test_split(
            pair_list, test_size=(val_size + test_size), random_state=self.seed
        )
        val_pairs, test_pairs = train_test_split(
            temp, test_size=(test_size / (val_size + test_size)), random_state=self.seed
        )
        
        print(f"  Train: {len(train_pairs)} ({len(train_pairs)/len(pair_list)*100:.1f}%)")
        print(f"  Val:   {len(val_pairs)} ({len(val_pairs)/len(pair_list)*100:.1f}%)")
        print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/len(pair_list)*100:.1f}%)")
        
        # Process splits
        split_data = {
            'train': (train_pairs, True),
            'val': (val_pairs, False),
            'test': (test_pairs, False)
        }
        
        all_metadata = {'train': {}, 'val': {}, 'test': {}}
        
        print("\n[Step 3/4] Processing images...")
        for split_idx, (split_name, (split_pair_list, do_augment)) in enumerate(split_data.items(), 1):
            print(f"\n[Step 3.{split_idx}/3] Processing {split_name.upper()} split...")
            
            augmentation = self.create_augmentation_pipeline(
                augment=(do_augment and augment_training)
            )
            
            processed_count = 0
            for image_stem in tqdm(split_pair_list, desc=f"Processing {split_name}"):
                orig_path, mask_path = pairs[image_stem]
                image, mask, valid = self.load_and_validate_pair(image_stem, orig_path, mask_path)
                
                if not valid:
                    continue
                
                # Preprocess
                image = self.normalize_image(image)
                mask = self.normalize_mask(mask)
                image, mask = self.resize_image_and_mask(image, mask)
                
                # Augment
                augmented = augmentation(image=image, mask=mask)
                image_aug = augmented['image']
                mask_aug = augmented['mask']
                
                # Save locally
                split_dir = self.output_dir / split_name
                image_path = split_dir / 'images' / f'{image_stem}.npy'
                mask_path = split_dir / 'masks' / f'{image_stem}.npy'
                
                np.save(image_path, image_aug.astype(np.float32))
                np.save(mask_path, mask_aug.astype(np.float32))
                
                # Store metadata
                all_metadata[split_name][image_stem] = {
                    'original_shape': image.shape,
                    'processed_shape': image_aug.shape,
                    'mask_pixels': float(np.sum(mask_aug)),
                    'mask_percentage': float(np.sum(mask_aug) / mask_aug.size * 100)
                }
                
                processed_count += 1
            
            print(f"  ✓ {processed_count} images processed for {split_name}")
        
        # Save metadata
        print("\n[Step 4/4] Saving metadata...")
        metadata_file = self.output_dir / 'metadata' / 'dataset_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Print summary
        self.print_summary()
        
        # Cleanup temp files
        self.cleanup_temp()
        
        return all_metadata
    
    def print_summary(self):
        """Print preprocessing summary."""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"✓ Total images found: {self.stats['total_images']}")
        print(f"✓ Valid pairs: {self.stats['valid_pairs']}")
        print(f"✗ Invalid pairs: {len(self.stats['invalid_pairs'])}")
        print(f"✓ Output directory: {self.output_dir}")
        print("="*60)
    
    def cleanup_temp(self):
        """Clean up temporary directories."""
        import shutil
        try:
            if self.temp_original and self.temp_original.exists():
                shutil.rmtree(self.temp_original)
            if self.temp_masks and self.temp_masks.exists():
                shutil.rmtree(self.temp_masks)
            print("✓ Temporary files cleaned up")
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")


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
        
        if not image_files:
            raise ValueError(f"No images found in {split} split!")
        
        print(f"Loading {len(image_files)} {split} samples...")
        
        def load_image_mask(image_path):
            image = np.load(image_path)
            mask_path = str(image_path).replace('/images/', '/masks/')
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


print("✓ Preprocessor classes defined!")

# ============================================================================
# CELL 4: Define U-Net Model Classes
# ============================================================================

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
        self.conv1 = self.conv_block(filters_base)
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv2 = self.conv_block(filters_base * 2)
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv3 = self.conv_block(filters_base * 4)
        self.pool3 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        self.conv4 = self.conv_block(filters_base * 8)
        self.pool4 = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
        
        # Bottleneck
        self.conv5 = self.conv_block(filters_base * 16)
        
        # Decoder
        self.upconv4 = layers.Conv2DTranspose(filters_base * 8, 3, strides=2, padding='same', activation='relu')
        self.conv6 = self.conv_block(filters_base * 8)
        
        self.upconv3 = layers.Conv2DTranspose(filters_base * 4, 3, strides=2, padding='same', activation='relu')
        self.conv7 = self.conv_block(filters_base * 4)
        
        self.upconv2 = layers.Conv2DTranspose(filters_base * 2, 3, strides=2, padding='same', activation='relu')
        self.conv8 = self.conv_block(filters_base * 2)
        
        self.upconv1 = layers.Conv2DTranspose(filters_base, 3, strides=2, padding='same', activation='relu')
        self.conv9 = self.conv_block(filters_base)
        
        # Output
        self.output_conv = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')
    
    def conv_block(self, filters):
        """Convolutional block."""
        return models.Sequential([
            layers.Conv2D(filters, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Conv2D(filters, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])
    
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
    
    def save(self, filepath):
        """Save model."""
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")


print("✓ Model classes defined!")

# ============================================================================
# CELL 5: Preprocess Dataset from ZIP
# ============================================================================

print("\n" + "="*70)
print("STARTING PREPROCESSING")
print("="*70)

preprocessor = ServerDatasetPreprocessor(
    original_zip=ORIGINAL_IMAGES_ZIP,
    masks_zip=MASKS_ZIP,
    output_dir=OUTPUT_DIR,
    target_size=(512, 512),
    seed=42
)

metadata = preprocessor.process_dataset(
    augment_training=True,
    val_size=0.15,
    test_size=0.10
)

print("\n✓ Preprocessing complete!")

# ============================================================================
# CELL 6: Load Data for Training
# ============================================================================

print("\n" + "="*70)
print("LOADING DATASETS")
print("="*70)

data_loader = DataLoader(dataset_dir=OUTPUT_DIR, batch_size=32)

train_dataset = data_loader.create_tf_dataset(split='train', shuffle=True)
val_dataset = data_loader.create_tf_dataset(split='val', shuffle=False)
test_dataset = data_loader.create_tf_dataset(split='test', shuffle=False)

print("✓ Datasets loaded successfully!")

# ============================================================================
# CELL 7: Build and Compile Model
# ============================================================================

print("\n" + "="*70)
print("BUILDING MODEL")
print("="*70)

model = SegmentationModel(input_shape=(512, 512, 3), num_classes=1)
model.build(filters_base=32)
model.compile(learning_rate=0.001)

# ============================================================================
# CELL 8: Train Model
# ============================================================================

print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)
print("Training for 50 epochs...")
print("(Check segmentation_results/logs/ for training history)\n")

history = model.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50
)

print("\n✓ Training complete!")

# ============================================================================
# CELL 9: Evaluate on Test Set
# ============================================================================

print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

model.evaluate(test_dataset)

# ============================================================================
# CELL 10: Save Model Locally
# ============================================================================

print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model_path = f'{OUTPUT_DIR}/models/construction_site_segmentation.h5'
Path(f'{OUTPUT_DIR}/models').mkdir(parents=True, exist_ok=True)
model.save(model_path)

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history.get('loss', [])],
    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
    'iou': [float(x) for x in history.history.get('iou', [])],
    'val_iou': [float(x) for x in history.history.get('val_iou', [])],
}

Path(f'{OUTPUT_DIR}/logs').mkdir(parents=True, exist_ok=True)
history_path = f'{OUTPUT_DIR}/logs/training_history.json'
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)

print(f"\n✓ SUCCESS!")
print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"\nFiles created:")
print(f"  Model: {os.path.abspath(model_path)}")
print(f"  History: {os.path.abspath(history_path)}")
print(f"  Preprocessed data:")
print(f"    Train: {os.path.abspath(f'{OUTPUT_DIR}/train')}")
print(f"    Val:   {os.path.abspath(f'{OUTPUT_DIR}/val')}")
print(f"    Test:  {os.path.abspath(f'{OUTPUT_DIR}/test')}")
print(f"\nNext steps:")
print(f"  1. Download segmentation_results/ folder to your laptop")
print(f"  2. Use the model for inference on new images")
print(f"  3. Check training_history.json for training curves")

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)