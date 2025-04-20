import os
import shutil

import albumentations as A
import cv2  # Import OpenCV
import torch
from albumentations.pytorch import ToTensorV2
from models.unet_maxvit import UNet
from torch import nn, optim
from torch.optim import lr_scheduler # Import LR scheduler (LinearLR, CosineAnnealingLR, SequentialLR)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming FieldSegmentationDataset is defined in utils.dataset and UNet in models.unet_maxvit
# Adjust imports based on your actual project structure if different
from utils.dataset import FieldSegmentationDataset  # Corrected class name


# --- Dice Loss 実装 ---
def dice_coeff(pred, target, smooth=1.0, epsilon=1e-6):
    """Calculates Dice Coefficient per class."""
    # pred: (N, C, H, W), target: (N, C, H, W)
    # Apply sigmoid to predictions
    pred = torch.sigmoid(pred)

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # (N, C, H*W)
    target_flat = target.view(target.shape[0], target.shape[1], -1)  # (N, C, H*W)

    intersection = (pred_flat * target_flat).sum(2)  # (N, C)
    pred_sum = pred_flat.sum(2)  # (N, C)
    target_sum = target_flat.sum(2)  # (N, C)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth + epsilon)  # (N, C)

    return dice  # Return per-class dice score for the batch


def dice_loss(pred, target, smooth=1.0, epsilon=1e-6):
    """Calculates Dice Loss (average over classes)."""
    dice_coeffs = dice_coeff(pred, target, smooth, epsilon)  # (N, C)
    # Average dice score across classes, then subtract from 1
    return 1.0 - dice_coeffs.mean()


def train_model(
    model,
    dataloader,
    num_epochs=500,
    device="cuda",
    bce_weight=0.5,
    dice_weight=0.5,
    initial_lr=4e-4, # Add initial_lr parameter
    warmup_epochs=10, # Add warmup_epochs parameter
    cosine_decay_epochs=500,
    min_lr = 1e-6,
    lr_scheduler_t_max=500, # Add lr_scheduler_t_max parameter
    lr_scheduler_eta_min=3e-5, # Add lr_scheduler_eta_min parameter
):
    """
    Trains the U-Net model using BCE + Dice loss with Linear Warmup + Cosine Decay LR schedule.

    Args:
        model (nn.Module): The U-Net model to train.
        dataloader (DataLoader): DataLoader providing training images and masks.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').
        bce_weight (float): Weight for BCE loss.
        dice_weight (float): Weight for Dice loss.
        initial_lr (float): Initial learning rate for the optimizer.
        warmup_epochs (int): Number of epochs for linear warmup.
        lr_scheduler_eta_min (float): Minimum learning rate for CosineAnnealingLR.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, switching to CPU.")
        device = "cpu"
    model.to(device)

    # Use BCEWithLogitsLoss (reduction='mean' is simpler here)
    criterion_bce = nn.BCEWithLogitsLoss()  # Default reduction='mean'
    # No need for separate Dice criterion instance if using the function directly

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-2) # Use initial_lr
    scaler = torch.amp.GradScaler("cuda")

    # --- Setup Linear Warmup + Cosine Decay Scheduler ---
    if warmup_epochs >= num_epochs:
        print(f"Warning: warmup_epochs ({warmup_epochs}) >= num_epochs ({num_epochs}). Using only LinearLR.")
        # Only linear warmup if warmup covers all epochs
        scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_epochs)
        scheduler = scheduler_warmup # Use only warmup scheduler
    elif warmup_epochs > 0:
        scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_decay_epochs, eta_min=min_lr,last_epoch=-1)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    # --- End Scheduler Setup ---

    print(
        f"Starting training on {device} for {num_epochs} epochs (BCE weight: {bce_weight}, Dice weight: {dice_weight}, Initial LR: {initial_lr})..."
    )
    # Scheduler type is printed during setup above
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            if batch is None:
                print("Warning: Skipping empty batch.")
                continue

            # Modify dataset __getitem__ to return filename if needed for safer inference later
            # Assuming batch now contains: imgs, masks (and potentially filename)
            # Example: imgs, masks, _ = batch # If filename is returned
            imgs, masks, filenames = batch  # Assuming original return format for now

            imgs = imgs.to(device)
            # Ensure masks are FloatTensor for loss functions and scale to 0.0-1.0
            masks = masks.to(device, dtype=torch.float) / 255.0

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                # Forward pass
                outputs = model(imgs)

                # --- Shape Checks (Unchanged, still good practice) ---
                if outputs.shape[2:] != masks.shape[2:]:
                    raise ValueError(f"Spatial dimension mismatch! Output: {outputs.shape}, Mask: {masks.shape}")
                if outputs.shape[1] != masks.shape[1]:
                    raise ValueError(f"Channel dimension mismatch! Output: {outputs.shape}, Mask: {masks.shape}")
                # --- End Shape Checks ---

                # --- Loss Calculation ---
                # Calculate BCE loss (averaged over batch and pixels)
                loss_bce = criterion_bce(outputs, masks)

                # Calculate Dice loss (averaged over classes)
                loss_dice = dice_loss(outputs, masks)  # Pass raw logits to dice_loss

                # Combine losses
                total_loss = bce_weight * loss_bce + dice_weight * loss_dice
                # --- End Loss Calculation ---

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # total_loss,loss_bce,loss_dixeのうちnanが出た場合は,ファイル名を出力
            if torch.isnan(total_loss) or torch.isnan(loss_bce) or torch.isnan(loss_dice):
                print(f"NaN loss encountered in epoch {epoch + 1}.")
                print(f"{filenames=}")
                print(f"  Loss values: total={total_loss.item():.4f}, bce={loss_bce.item():.4f}, dice={loss_dice.item():.4f}")
                # --- Debug Info ---
                print("  --- Debugging Tensor Stats ---")
                print(f"  Outputs (model predictions) stats:")
                print(f"    Shape: {outputs.shape}")
                print(f"    Contains NaN: {torch.isnan(outputs).any().item()}") # OutputsにNaNが含まれているのが原因
                print(f"    Contains Inf: {torch.isinf(outputs).any().item()}")
                if not torch.isnan(outputs).any() and not torch.isinf(outputs).any():
                    print(f"    Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}, Mean: {outputs.mean().item():.4f}, Std: {outputs.std().item():.4f}")
                print(f"  Masks (target) stats:")
                print(f"    Shape: {masks.shape}")
                print(f"    Contains NaN: {torch.isnan(masks).any().item()}")
                print(f"    Contains Inf: {torch.isinf(masks).any().item()}")
                if not torch.isnan(masks).any() and not torch.isinf(masks).any():
                    print(f"    Min: {masks.min().item():.4f}, Max: {masks.max().item():.4f}, Mean: {masks.mean().item():.4f}, Std: {masks.std().item():.4f}")
                print("  -----------------------------")
                # Assuming filename is part of the batch - requires dataset modification
                # print(f"Filename: {batch[2]}") # Uncomment if filename is returned in batch
                continue

            running_loss += total_loss.item()
            running_bce_loss += loss_bce.item()
            running_dice_loss += loss_dice.item()
            progress_bar.set_postfix(loss=total_loss.item(), bce=loss_bce.item(), dice=loss_dice.item())

        avg_loss = running_loss / len(dataloader)
        avg_bce_loss = running_bce_loss / len(dataloader)
        avg_dice_loss = running_dice_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} (BCE: {avg_bce_loss:.4f}, Dice: {avg_dice_loss:.4f}) LR: {current_lr:.6f}")

        # Step the scheduler after each epoch
        scheduler.step()

    print("Training finished.")


if __name__ == "__main__":
    # --- Configuration ---
    ROOT_DIR = "/workspace/projects/solafune-field-area-segmentation"
    EX_NUM = "ex1"  # Example experiment number
    IMAGE_DIR = os.path.join(ROOT_DIR, "data/train_images")  # Path to training images (adjust if needed)
    ANNOTATION_FILE = os.path.join(
        ROOT_DIR, "data/train_annotation.json"
    )  # Path to training annotations (adjust if needed)
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", EX_NUM, "check")  # Path to save model outputs
    BACKBONE = "maxvit_small_tf_512.in1k"  # Example backbone
    NUM_OUTPUT_CHANNELS = 3  # Number of output channels (field, edge, contact)
    PRETRAINED = True
    BATCH_SIZE = 2  # Adjust based on GPU memory
    NUM_WORKERS = 4  # Adjust based on CPU cores
    NUM_EPOCHS = 500  # Number of training epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_H = 512  # Example, not directly used if RandomCrop is applied
    INPUT_W = 512  # Example, not directly used if RandomCrop is applied
    SCALE_FACTOR = 3  # Resize scale factor for initial loading in dataset
    CROP_H = 512  # Height after RandomCrop
    CROP_W = 512  # Width after RandomCrop
    RESIZE_H = 1024  # Height after Resize transform (model input)
    RESIZE_W = 1024  # Width after Resize transform (model input)
    # Pre-calculated mean/std (Example values - REPLACE WITH YOUR ACTUAL VALUES)
    # These should be calculated across your entire training dataset for all 12 channels
    # Example: DATASET_MEAN = [0.1, 0.1, ..., 0.1] # List of 12 means
    # Example: DATASET_STD = [0.05, 0.05, ..., 0.05] # List of 12 stds
    DATASET_MEAN = None  # Set to None to use per-image normalization if not pre-calculated
    DATASET_STD = None  # Set to None to use per-image normalization if not pre-calculated
    # Loss weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    # LR Scheduler settings
    LR_SCHEDULER_T_MAX = 50
    LR_SCHEDULER_ETA_MIN = 1e-6
    # ---------------------

    print("Setting up dataset and dataloader...")
    # Define transformations including the required resize
    # Note: Normalization is handled inside the Dataset class now

    # MaxViTモデルは入力サイズが16の倍数である必要があるため、それに合わせて調整
    # 各画像のサイズは異なるため、PadIfNeededを使用して16の倍数にパディング
    transform = A.Compose(
        [
            # 512x512の領域をrandom_crop
            A.RandomCrop(height=CROP_H, width=CROP_W, p=1.0),
            # Resize to 1024x1024
            A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_NEAREST,),
            # 16の倍数になるようにパディング（min_heightとmin_widthは16の倍数に切り上げ）
            # A.PadIfNeeded(
            #     min_height=16 * ((RESIZE_H + 15) // 16),
            #     min_width=16 * ((RESIZE_W + 15) // 16),
            #     border_mode=cv2.BORDER_CONSTANT
            # ),
            # Add other augmentations here if needed (e.g., Flip, Rotate)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),  # Converts image HWC->CHW, mask HWC->CHW, scales image 0-255 -> 0-1 (mask remains 0 or 255 uint8)
        ]
    )

    print(f"Images will be resized to {RESIZE_H}x{RESIZE_W} and padded to ensure dimensions are divisible by 16")

    # Save masks for debugging
    if os.path.exists(f"/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check"):
        try:
            # Remove existing check directory if it exists
            # 1. Remove existing directory
            shutil.rmtree(
                f"/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check", ignore_errors=True
            )
            os.makedirs(f"/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check", exist_ok=True)
        except Exception as e:
            print(f"Error removing existing check directory: {e}")
    else:
        # 2. Create new directory
        os.makedirs(f"/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check", exist_ok=True)
        print(f"Output directory created: {OUTPUT_DIR}")
    # Ensure FieldSegmentationDataset is correctly implemented and paths/file are valid
    try:
        # Initialize dataset with paths, mean/std, and transform
        dataset = FieldSegmentationDataset(
            img_dir=IMAGE_DIR,
            ann_json_path=ANNOTATION_FILE,  # Corrected parameter name
            scale_factor=SCALE_FACTOR,  # Pass scale_factor
            transform=transform,
            # edge_width and contact_width use defaults if not specified
            contact_width=5,
            edge_width=3,
            mean=DATASET_MEAN,  # Pass pre-calculated mean
            std=DATASET_STD,  # Pass pre-calculated std
        )

        if len(dataset) == 0:
            print(
                f"Error: Dataset is empty. Check image path '{IMAGE_DIR}' and annotation file '{ANNOTATION_FILE}', and ensure they contain matching, valid data."
            )
            # Exit if dataset is empty, as dummy data generation is complex here
            exit()
        print(f"Dataset initialized with {len(dataset)} samples.")
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE == "cuda" else False,
        )
        print(f"Dataset size: {len(dataset)}, Dataloader ready.")

        print("Initializing model...")
        # Initialize model with the number of output channels (not classes for BCE loss)
        model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS)
        model.to(DEVICE)  # Move model to the specified device
        # print(summary(model, (12, 512, 512)))
        print(f"Model: UNet with {BACKBONE} backbone, {NUM_OUTPUT_CHANNELS} output channels.")

        # Start training
        train_model(
            model,
            dataloader,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            bce_weight=BCE_WEIGHT,
            dice_weight=DICE_WEIGHT,
            lr_scheduler_t_max=LR_SCHEDULER_T_MAX,
            lr_scheduler_eta_min=LR_SCHEDULER_ETA_MIN,
        )

        # Save the model after training
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Output directory {OUTPUT_DIR} created.")
        else:
            print(f"Output directory {OUTPUT_DIR} already exists. Model will be saved there.")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.path"))
        print(f"Model saved to {OUTPUT_DIR}")

    except NameError:
        print(
            "Error: FieldSegmentationDataset or UNet class not found. Ensure 'src' is in PYTHONPATH or run from the project root. Cannot run training."
        )
    except FileNotFoundError as e:
        print(f"Error: File or directory not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during setup or training: {e}")
