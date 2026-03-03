import zipfile
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex

# ==============================
# UNZIP DATASET
# ==============================
zip_path = "archive.zip"

if not os.path.exists("dataset"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("dataset")

BASE_PATH = "dataset"

subfolders = os.listdir(BASE_PATH)
if len(subfolders) == 1:
    BASE_PATH = os.path.join(BASE_PATH, subfolders[0])

train_images = f"{BASE_PATH}/images/train"
train_masks  = f"{BASE_PATH}/masks/train"
val_images   = f"{BASE_PATH}/images/val"
val_masks    = f"{BASE_PATH}/masks/val"

print("Train images:", len(os.listdir(train_images)))
print("Train masks:", len(os.listdir(train_masks)))

# ==============================
# RGB → CLASS CONVERSION
# ==============================
# NOTE: You MUST adjust mapping based on dataset colors if needed
def rgb_to_class(mask):
    # If mask already single channel, return directly
    if len(mask.shape) == 2:
        return mask

    # Convert RGB colors to class indices (simple fallback)
    # This works if mask already encoded numerically in RGB
    return mask[:, :, 0]

# ==============================
# DATASET CLASS
# ==============================
class DroneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

        # safer mask matching
        self.masks = sorted(os.listdir(mask_dir))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = rgb_to_class(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# ==============================
# TRANSFORMS
# ==============================
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# ==============================
# DATALOADER
# ==============================
train_dataset = DroneDataset(train_images, train_masks, transform)
val_dataset   = DroneDataset(val_images, val_masks, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

# ==============================
# MODEL
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 24  # verify if needed

model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b3",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
)

model.to(DEVICE)

# ==============================
# LOSS + OPTIMIZER
# ==============================
loss_fn = smp.losses.DiceLoss(mode="multiclass") + nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

iou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

# ==============================
# VISUALIZATION FUNCTION
# ==============================
def visualize(image, mask, pred):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(image.permute(1,2,0).cpu())

    plt.subplot(1,3,2)
    plt.title("Mask")
    plt.imshow(mask.cpu())

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred.cpu())

    plt.show()

# ==============================
# TRAINING
# ==============================
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    iou_metric.reset()

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)

            # visualize first batch of each epoch
            if i == 0:
                visualize(images[0], masks[0], preds[0])

    iou = iou_metric.compute()

    scheduler.step(val_loss)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"IoU Score: {iou:.4f}")

# ==============================
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), "drone_segmentation.pth")
print("Model saved!")