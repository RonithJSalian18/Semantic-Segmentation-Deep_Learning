"""
SERVER TRAINING NOTEBOOK - SINGLE ZIP INPUT
Construction Site Segmentation (CUDA ENABLED)
"""

# ============================================================================
# CELL 1: Imports + GPU SETUP
# ============================================================================

import os, sys, cv2, json, zipfile, tempfile, warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
import tensorflow as tf

warnings.filterwarnings('ignore')

print("✓ Libraries imported")

# ================= GPU / CUDA CHECK =================
print("\n🔍 Checking GPU...")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ GPU Found: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU found, using CPU")

print("Num GPUs Available:", len(gpus))
print("Device Name:", tf.test.gpu_device_name())
# ===================================================


# ============================================================================
# CELL 2: CONFIG
# ============================================================================

DATA_ZIP = "./data.zip"
OUTPUT_DIR = "./segmentation_results"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

if not os.path.exists(DATA_ZIP):
    print(f"❌ ERROR: {DATA_ZIP} not found")
    sys.exit(1)

print(f"✓ Using ZIP: {os.path.abspath(DATA_ZIP)}")


# ============================================================================
# CELL 3: PREPROCESSOR
# ============================================================================

class ServerDatasetPreprocessor:

    def __init__(self, data_zip, output_dir, target_size=(512,512), seed=42):
        self.data_zip = Path(data_zip)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        np.random.seed(seed)

        self.create_output_structure()

        self.stats = {
            'total_images': 0,
            'valid_pairs': 0,
            'invalid_pairs': []
        }

    def create_output_structure(self):
        for split in ['train','val','test']:
            (self.output_dir/split/'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir/split/'masks').mkdir(parents=True, exist_ok=True)
        (self.output_dir/'metadata').mkdir(parents=True, exist_ok=True)

    def extract_zip(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dataset_"))
        with zipfile.ZipFile(self.data_zip,'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

        self.temp_original = self.temp_dir/"input"/"original_images"
        self.temp_masks = self.temp_dir/"input"/"masked_images"

        if not self.temp_original.exists() or not self.temp_masks.exists():
            raise ValueError("❌ Wrong ZIP structure")

        print("✓ ZIP extracted")

    def find_pairs(self):
        orig = {f.stem:f for f in self.temp_original.rglob("*") if f.suffix.lower() in ['.png','.jpg','.jpeg']}
        masks = {f.stem:f for f in self.temp_masks.rglob("*") if f.suffix.lower() in ['.png','.jpg','.jpeg']}

        pairs = {k:(orig[k],masks[k]) for k in orig if k in masks}
        self.stats['total_images'] = len(pairs)

        print(f"✓ Found {len(pairs)} pairs")
        return pairs

    def process_dataset(self):

        self.extract_zip()
        pairs = self.find_pairs()
        keys = list(pairs.keys())

        train,valtest = train_test_split(keys,test_size=0.25,random_state=42)
        val,test = train_test_split(valtest,test_size=0.4,random_state=42)

        splits = {'train':train,'val':val,'test':test}

        for split,items in splits.items():
            for k in tqdm(items, desc=split):
                img_path,mask_path = pairs[k]

                img = cv2.imread(str(img_path))
                mask = cv2.imread(str(mask_path),0)

                if img is None or mask is None:
                    continue

                img = cv2.resize(img,(512,512))/255.0
                mask = cv2.resize(mask,(512,512))
                mask = (mask>127).astype(np.float32)

                np.save(self.output_dir/split/'images'/f"{k}.npy",img.astype(np.float32))
                np.save(self.output_dir/split/'masks'/f"{k}.npy",mask.astype(np.float32))

        print("✓ Dataset processed")


# ============================================================================
# CELL 4: DATALOADER
# ============================================================================

class DataLoader:
    def __init__(self, dataset_dir, batch_size=16):
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size

    def create_tf_dataset(self, split='train'):
        import tensorflow as tf

        files = sorted((self.dataset_dir/split/'images').glob("*.npy"))

        def load(x):
            x = x.numpy().decode("utf-8")
            img = np.load(x)
            mask = np.load(x.replace("images", "masks"))
            return img.astype(np.float32), mask.astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices([str(f) for f in files])
        ds = ds.map(lambda x: tf.py_function(load,[x],[tf.float32,tf.float32]))

        def _fix_shape(img, mask):
            img.set_shape((512, 512, 3))
            mask.set_shape((512, 512))
            return img, mask

        ds = ds.map(_fix_shape)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


# ============================================================================
# CELL 5: MODEL
# ============================================================================

from tensorflow.keras import layers, models

def conv_block(x,f):
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    return x

def build_unet():
    inputs = layers.Input((512,512,3))

    c1 = conv_block(inputs,32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1,64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2,128); p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3,256)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1,c3])
    c4 = conv_block(u1,128)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2,c2])
    c5 = conv_block(u2,64)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3,c1])
    c6 = conv_block(u3,32)

    outputs = layers.Conv2D(1,1,activation='sigmoid')(c6)

    return models.Model(inputs,outputs)


# ============================================================================
# CELL 6: RUN PREPROCESSING
# ============================================================================

preprocessor = ServerDatasetPreprocessor(DATA_ZIP, OUTPUT_DIR)
preprocessor.process_dataset()


# ============================================================================
# CELL 7: LOAD DATA
# ============================================================================

loader = DataLoader(OUTPUT_DIR)

train_ds = loader.create_tf_dataset('train')
val_ds = loader.create_tf_dataset('val')
test_ds = loader.create_tf_dataset('test')


# ============================================================================
# CELL 8: TRAIN (GPU FORCED)
# ============================================================================

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    print(f"\n🚀 Training on {device}")

    model = build_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=20)


# ============================================================================
# CELL 9: EVALUATE + SAVE
# ============================================================================

model.evaluate(test_ds)

Path(f"{OUTPUT_DIR}/models").mkdir(parents=True, exist_ok=True)
model.save(f"{OUTPUT_DIR}/models/model.h5")

print("🎉 DONE!")