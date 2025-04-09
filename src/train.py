import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import cv2 # Import OpenCV
import os

from torchsummary import summary


# Assuming FieldSegmentationDataset is defined in utils.dataset and UNet in models.unet_maxvit
# Adjust imports based on your actual project structure if different
try:
    from utils.dataset import FieldSegmentationDataset # Corrected class name
    from models.unet_maxvit import UNet
except ImportError:
    print("Warning: Could not import CustomDataset or UNet. Ensure they are defined in the correct paths (src/utils/dataset.py and src/models/unet_maxvit.py)")
    # Define dummy classes if imports fail, to allow the script to load
    class FieldSegmentationDataset: # Corrected dummy class name
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 0
        def __getitem__(self, idx): return torch.zeros(3, 64, 64), torch.zeros(3, 64, 64) # Return dummy tensors
    class UNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Linear(1,1)
        def forward(self, x): return self.dummy(torch.zeros(x.shape[0], 1))


def train_model(model, dataloader, num_epochs=10, device='cuda'):
    """
    Trains the U-Net model.

    Args:
        model (nn.Module): The U-Net model to train.
        dataloader (DataLoader): DataLoader providing training images and masks.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, switching to CPU.")
        device = 'cpu'
    model.to(device)
    # Use BCEWithLogitsLoss for multi-label segmentation (each channel independently)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            # Handle potential errors if dataset returns None (e.g., dummy data case)
            if batch is None:
                print("Warning: Skipping empty batch.")
                continue
            imgs, masks = batch

            # バッチの形状をログに出力
            print(f"Batch shapes - imgs: {imgs.shape}, masks: {masks.shape}")

            # Ensure masks are FloatTensor for BCEWithLogitsLoss and scale to 0.0-1.0
            # Dataset now returns uint8 [0, 255], convert to float [0.0, 1.0]
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.float) / 255.0
            

            # Zero the parameter gradients
            optimizer.zero_grad()
            print("====================================")

            # Forward pass
            outputs = model(imgs)
            print("-----------------")
            
            # モデルの出力形状とマスクの形状を損失計算直前に再度確認
            print(f"[Before Loss] Model output shape: {outputs.shape}, dtype: {outputs.dtype}")
            print(f"[Before Loss] Masks shape: {masks.shape}, dtype: {masks.dtype}")

            # Ensure output and target shapes match for BCEWithLogitsLoss: (N, C, H, W)
            # Check spatial dimensions explicitly
            if outputs.shape[2:] != masks.shape[2:]:
                 print(f"Error: Spatial dimensions mismatch! Output: {outputs.shape[2:]}, Mask: {masks.shape[2:]}")
                 # Optionally raise an error or handle it
                 raise ValueError(f"Spatial dimension mismatch between model output {outputs.shape} and mask {masks.shape}")
            # Check channel dimension (should be handled by the error message, but good to be explicit)
            if outputs.shape[1] != masks.shape[1]:
                 print(f"Error: Channel dimensions mismatch! Output: {outputs.shape[1]}, Mask: {masks.shape[1]}")
                 raise ValueError(f"Channel dimension mismatch between model output {outputs.shape} and mask {masks.shape}")

            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    # --- Configuration ---
    ROOT_DIR = '/workspace/projects/solafune-field-area-segmentation'
    IMAGE_DIR = os.path.join(ROOT_DIR, 'data/train_images_mini') # Path to training images (adjust if needed)
    ANNOTATION_FILE = os.path.join(ROOT_DIR, 'data/train_annotation.json') # Path to training annotations (adjust if needed)
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs','ex0') # Path to save model outputs
    BACKBONE = 'maxvit_small_tf_512.in1k' # Example backbone
    NUM_OUTPUT_CHANNELS = 3 # Number of output channels (field, edge, contact)
    PRETRAINED = True
    BATCH_SIZE = 4 # Adjust based on GPU memory
    NUM_WORKERS = 4 # Adjust based on CPU cores
    NUM_EPOCHS = 2 # Number of training epochs
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_H = 512 # Original image height (example)
    INPUT_W = 512 # Original image width (example)
    SCALE_FACTOR = 1 # Resize scale factor from requirements
    RESIZE_H = INPUT_H * SCALE_FACTOR
    RESIZE_W = INPUT_W * SCALE_FACTOR
    # Pre-calculated mean/std (Example values - REPLACE WITH YOUR ACTUAL VALUES)
    # These should be calculated across your entire training dataset for all 12 channels
    # Example: DATASET_MEAN = [0.1, 0.1, ..., 0.1] # List of 12 means
    # Example: DATASET_STD = [0.05, 0.05, ..., 0.05] # List of 12 stds
    DATASET_MEAN = None # Set to None to use per-image normalization if not pre-calculated
    DATASET_STD = None  # Set to None to use per-image normalization if not pre-calculated
    # ---------------------

    print("Setting up dataset and dataloader...")
    # Define transformations including the required resize
    # Note: Normalization is handled inside the Dataset class now
    
    # MaxViTモデルは入力サイズが16の倍数である必要があるため、それに合わせて調整
    # 各画像のサイズは異なるため、PadIfNeededを使用して16の倍数にパディング
    transform = A.Compose([
        # まず指定サイズにリサイズ
        A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_LINEAR),
        # 16の倍数になるようにパディング（min_heightとmin_widthは16の倍数に切り上げ）
        A.PadIfNeeded(
            min_height=16 * ((RESIZE_H + 15) // 16),
            min_width=16 * ((RESIZE_W + 15) // 16),
            border_mode=cv2.BORDER_CONSTANT
        ),
        # Add other augmentations here if needed (e.g., Flip, Rotate)
        # A.HorizontalFlip(p=0.5),
        ToTensorV2(), # Converts image HWC->CHW, mask HWC->CHW, scales image 0-255 -> 0-1 (mask remains 0 or 255 uint8)
    ])
    
    print(f"Images will be resized to {RESIZE_H}x{RESIZE_W} and padded to ensure dimensions are divisible by 16")

    # Ensure FieldSegmentationDataset is correctly implemented and paths/file are valid
    try:
        # Initialize dataset with paths, mean/std, and transform
        dataset = FieldSegmentationDataset(
            img_dir=IMAGE_DIR,
            ann_json_path=ANNOTATION_FILE, # Corrected parameter name
            transform=transform,
            mean=DATASET_MEAN, # Pass pre-calculated mean
            std=DATASET_STD   # Pass pre-calculated std
            # edge_width and contact_width use defaults if not specified
        )

        if len(dataset) == 0:
             print(f"Error: Dataset is empty. Check image path '{IMAGE_DIR}' and annotation file '{ANNOTATION_FILE}', and ensure they contain matching, valid data.")
             # Exit if dataset is empty, as dummy data generation is complex here
             exit()
        print(f"Dataset initialized with {len(dataset)} samples.")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if DEVICE == 'cuda' else False)
        print(f"Dataset size: {len(dataset)}, Dataloader ready.")

        print("Initializing model...")
        # Initialize model with the number of output channels (not classes for BCE loss)
        model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS)
        model.to(DEVICE)  # Move model to the specified device
        print(summary(model, (12, 512, 512)))
        print(f"Model: UNet with {BACKBONE} backbone, {NUM_OUTPUT_CHANNELS} output channels.")

        # Start training
        train_model(model, dataloader, num_epochs=NUM_EPOCHS, device=DEVICE)
        
        # Save the model after training
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Output directory {OUTPUT_DIR} created.")
        else:
            print(f"Output directory {OUTPUT_DIR} already exists. Model will be saved there.")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,'model.path'))
        print(f"Model saved to {OUTPUT_DIR}")

    except NameError:
         print("Error: FieldSegmentationDataset or UNet class not found. Ensure 'src' is in PYTHONPATH or run from the project root. Cannot run training.")
    except FileNotFoundError as e:
         print(f"Error: File or directory not found. Please check paths. Details: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during setup or training: {e}")