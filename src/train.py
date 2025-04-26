import os
import shutil
import json # アノテーションファイル読み込みのため追加
import numpy as np
import torch
import torchvision # For image logging
import wandb # Import WANDB
from dotenv import load_dotenv # Import dotenv

import albumentations as A
import cv2  # Import OpenCV
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
    train_dataloader, # Renamed from dataloader
    valid_dataloader, # Added for validation
    validation_interval, # Added for validation frequency
    num_epochs=500,
    device="cuda",
    bce_weight=0.5,
    dice_weight=0.5,
    initial_lr=4e-4, # Add initial_lr parameter
    warmup_epochs=10, # Add warmup_epochs parameter
    cosine_decay_epochs=500,
    min_lr = 1e-6,
    lr_scheduler_t_max=500, # Add lr_scheduler_t_max parameter - Note: This might need adjustment based on train epochs only
    lr_scheduler_eta_min=3e-5, # Add lr_scheduler_eta_min parameter
):
    """
    Trains the U-Net model using BCE + Dice loss with Linear Warmup + Cosine Decay LR schedule,
    and performs validation periodically.

    Args:
        model (nn.Module): The U-Net model to train.
        train_dataloader (DataLoader): DataLoader providing training images and masks.
        valid_dataloader (DataLoader): DataLoader providing validation images and masks (can be None).
        validation_interval (int): How often to run validation (in epochs).
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').
        bce_weight (float): Weight for BCE loss.
        dice_weight (float): Weight for Dice loss.
        initial_lr (float): Initial learning rate for the optimizer.
        warmup_epochs (int): Number of epochs for linear warmup.
        cosine_decay_epochs (int): Total epochs for cosine decay (often num_epochs - warmup_epochs).
        min_lr (float): Minimum learning rate for cosine decay.
        lr_scheduler_t_max (int): T_max for CosineAnnealingLR (often num_epochs - warmup_epochs).
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
    scaler = torch.amp.GradScaler(device) # Use device string directly

    # --- Setup Linear Warmup + Cosine Decay Scheduler ---
    # Adjust cosine decay epochs based on warmup
    actual_cosine_epochs = num_epochs - warmup_epochs
    if actual_cosine_epochs <= 0:
        print(f"Warning: warmup_epochs ({warmup_epochs}) >= num_epochs ({num_epochs}). Cosine decay part will not run.")
        actual_cosine_epochs = 1 # Avoid T_max=0 error

    if warmup_epochs >= num_epochs:
        print(f"Warning: warmup_epochs ({warmup_epochs}) >= num_epochs ({num_epochs}). Using only LinearLR.")
        # Only linear warmup if warmup covers all epochs
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_epochs)
    elif warmup_epochs > 0:
        scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        # Use adjusted cosine epochs for T_max
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_cosine_epochs, eta_min=min_lr)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    else: # No warmup
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
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)

        for batch in progress_bar:
            if batch is None:
                print("Warning: Skipping empty batch.")
                continue

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
                continue

            running_loss += total_loss.item()
            running_bce_loss += loss_bce.item()
            running_dice_loss += loss_dice.item()
            progress_bar.set_postfix(loss=total_loss.item(), bce=loss_bce.item(), dice=loss_dice.item())

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_bce_loss = running_bce_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_dice_loss = running_dice_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} Avg Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce_loss:.4f}, Dice: {avg_train_dice_loss:.4f}) LR: {current_lr:.6f}")

        # --- Log Training Metrics to WANDB ---
        if wandb.run:
            wandb.log({
                "epoch": epoch + 1, # Log epoch number (1-based)
                "avg_train_loss": avg_train_loss,
                "avg_train_bce_loss": avg_train_bce_loss,
                "avg_train_dice_loss": avg_train_dice_loss,
                "learning_rate": current_lr,
            }) # Logged against the epoch step implicitly by wandb
        # --- End Log Training Metrics ---

        # Step the scheduler after each epoch
        scheduler.step()

        # --- Validation Phase ---
        if (epoch + 1) % validation_interval == 0:
            model.eval()
            running_val_loss = 0.0
            running_val_bce_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_dice_coeff = 0.0 # Initialize Dice Coeff accumulator
            num_val_samples = 0 # Initialize validation sample counter
            # Check if valid_dataloader is not None and has items before creating tqdm
            if valid_dataloader and len(valid_dataloader) > 0:
                val_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]", leave=False)

                with torch.no_grad():
                    for val_batch in val_progress_bar:
                        if val_batch is None: continue # Skip empty batches if any

                        val_imgs, val_masks, _ = val_batch # Assuming filename is returned but not needed here
                        val_imgs = val_imgs.to(device)
                        val_masks = val_masks.to(device, dtype=torch.float) / 255.0

                        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                            val_outputs = model(val_imgs)

                            # --- Shape Checks ---
                            if val_outputs.shape[2:] != val_masks.shape[2:]:
                                print(f"Warning: Validation shape mismatch! Output: {val_outputs.shape}, Mask: {val_masks.shape}")
                                continue # Skip this batch if shapes mismatch
                            if val_outputs.shape[1] != val_masks.shape[1]:
                                print(f"Warning: Validation channel mismatch! Output: {val_outputs.shape}, Mask: {val_masks.shape}")
                                continue # Skip this batch if shapes mismatch
                            # --- End Shape Checks ---

                            val_loss_bce = criterion_bce(val_outputs, val_masks)
                            val_loss_dice = dice_loss(val_outputs, val_masks)
                            val_total_loss = bce_weight * val_loss_bce + dice_weight * val_loss_dice

                        if not torch.isnan(val_total_loss): # Only accumulate if not NaN
                            running_val_loss += val_total_loss.item()
                            running_val_bce_loss += val_loss_bce.item()
                            running_val_dice_loss += val_loss_dice.item()
                            val_progress_bar.set_postfix(loss=val_total_loss.item(), bce=val_loss_bce.item(), dice=val_loss_dice.item())
                        else:
                            print(f"Warning: NaN validation loss encountered in epoch {epoch + 1}.")
                            continue # Skip dice calculation if loss is NaN

                        # --- Calculate Dice Coefficient for the batch ---
                        with torch.no_grad(): # Ensure no gradients are computed here
                            # Apply sigmoid to outputs for dice calculation
                            val_outputs_sigmoid = torch.sigmoid(val_outputs)
                            # Calculate dice coefficient using the existing dice_coeff function
                            # dice_coeff returns per-class dice: (N, C)
                            dice_coeffs_batch = dice_coeff(val_outputs_sigmoid, val_masks, smooth=1.0, epsilon=1e-6) # Pass sigmoid output
                            # Average over classes for each item in the batch -> (N,)
                            dice_per_item = dice_coeffs_batch.mean(dim=1)
                            # Sum the dice scores for the batch
                            running_val_dice_coeff += dice_per_item.sum().item()
                            num_val_samples += val_imgs.size(0) # Count samples processed
                        # --- End Calculate Dice Coefficient ---


                # Avoid division by zero
                avg_val_loss = running_val_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_bce_loss = running_val_bce_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_dice_loss = running_val_dice_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_dice = running_val_dice_coeff / num_val_samples if num_val_samples > 0 else 0.0 # Calculate avg Dice Coeff
                print(f"Epoch {epoch + 1} Avg Valid Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce_loss:.4f}, Dice: {avg_val_dice_loss:.4f}) Dice Coeff: {avg_val_dice:.4f}") # Add Dice Coeff to print

                # --- Log Validation Metrics and Images to WANDB ---
                if wandb.run:
                    log_dict = {
                        "epoch": epoch + 1, # Log against epoch
                        "avg_val_loss": avg_val_loss,
                        "avg_val_bce_loss": avg_val_bce_loss,
                        "avg_val_dice_loss": avg_val_dice_loss,
                        "avg_val_dice_coeff": avg_val_dice,
                    }

                    # --- Log Validation Images ---
                    log_images = []
                    # Use the last batch's data (val_imgs, val_masks, val_outputs)
                    if 'val_imgs' in locals() and val_imgs is not None: # Check if variables exist from the loop
                        num_samples_to_log = min(4, val_imgs.size(0)) # Log up to 4 samples

                        # Ensure tensors are on CPU and detached for processing
                        val_imgs_cpu = val_imgs[:num_samples_to_log].cpu().detach()
                        val_masks_cpu = val_masks[:num_samples_to_log].cpu().detach()
                        val_outputs_cpu = val_outputs[:num_samples_to_log].cpu().detach()

                        # Apply sigmoid and threshold for prediction mask visualization
                        preds_sigmoid = torch.sigmoid(val_outputs_cpu)
                        preds_binary = (preds_sigmoid > 0.5).float() # Threshold at 0.5

                        for i in range(num_samples_to_log):
                            # Input image: (C, H, W), assume C=3
                            img = val_imgs_cpu[i]
                            # Ground Truth Mask: (C, H, W), assume C=3 (field, edge, contact)
                            mask_gt_combined = val_masks_cpu[i] # Already (3, H, W)
                            # Prediction Mask: (C, H, W), same format
                            mask_pred_combined = preds_binary[i] # Already (3, H, W)

                            # Clamp values to avoid warnings if slightly outside [0,1]
                            # img: (12, H, W) -> select channels 2,3,4 for visualization
                            img = img[[2, 3, 4], ...]
                            img = torch.clamp(img, 0, 1)
                            mask_gt_combined = torch.clamp(mask_gt_combined, 0, 1)
                            mask_pred_combined = torch.clamp(mask_pred_combined, 0, 1)

                            # Create a grid: [Image | GT Mask | Pred Mask]
                            combined_image = torchvision.utils.make_grid(
                                [img, mask_gt_combined, mask_pred_combined],
                                nrow=3, padding=2, normalize=False # Already in [0,1] range
                            )
                            log_images.append(wandb.Image(combined_image,
                                            caption=f"Epoch {epoch+1} Sample {i} (Img | GT | Pred)"))

                        if log_images: # Only add if images were generated
                            log_dict["validation_samples"] = log_images
                    # --- End Log Validation Images ---

                    # Log all validation metrics and images for this epoch
                    wandb.log(log_dict)
                # --- End Log Validation Metrics and Images ---
            else:
                print(f"Epoch {epoch + 1} Validation skipped (empty or no dataloader).")

            model.train() # Switch back to training mode
        # --- End Validation Phase ---

    print("Training finished.")


if __name__ == "__main__":
    # --- Load Environment Variables ---
    load_dotenv()
    # --- End Load Environment Variables ---

    # --- Configuration ---
    ROOT_DIR = "/workspace/projects/solafune-field-area-segmentation"
    EX_NUM = "ex5"  # Example experiment number
    IMAGE_DIR = os.path.join(ROOT_DIR, "data/train_images")  # Path to training images (adjust if needed)
    ANNOTATION_FILE = os.path.join(
        ROOT_DIR, "data/train_annotation.json"
    )  # Path to training annotations (adjust if needed)
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", EX_NUM, "check")  # Path to save model outputs
    CACHE_DIR = os.path.join(ROOT_DIR, "outputs", EX_NUM, "cache")  # Path to save cache files
    BACKBONE = "maxvit_small_tf_512.in1k"  # Example backbone
    NUM_OUTPUT_CHANNELS = 3  # Number of output channels (field, edge, contact)
    PRETRAINED = True
    BATCH_SIZE = 2  # Adjust based on GPU memory
    NUM_WORKERS = 4  # Adjust based on CPU cores
    NUM_EPOCHS = 1000  # Number of training epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_H = 512  # Example, not directly used if RandomCrop is applied
    INPUT_W = 512  # Example, not directly used if RandomCrop is applied
    SCALE_FACTOR = 2 # Resize scale factor for initial loading in dataset
    CROP_H = 800  # Height after RandomCrop
    CROP_W = 800  # Width after RandomCrop
    RESIZE_H = 1024  # Height after Resize transform (model input)
    RESIZE_W = 1024  # Width after Resize transform (model input)
    # Pre-calculated mean/std (Example values - REPLACE WITH YOUR ACTUAL VALUES)
    DATASET_MEAN = None  # Set to None to use per-image normalization if not pre-calculated
    DATASET_STD = None  # Set to None to use per-image normalization if not pre-calculated
    # Loss weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    # LR Scheduler settings
    INITIAL_LR = 4e-4 # Define initial LR here
    WARMUP_EPOCHS = 10 # Define warmup epochs here
    MIN_LR = 1e-6 # Define min LR here
    # LR_SCHEDULER_T_MAX = 50 # This might be redundant if calculated inside train_model
    # LR_SCHEDULER_ETA_MIN = 1e-6 # This is now min_lr

    # Validation settings
    VALID_IMG_INDEX = [0, 5, 10, 15, 20] # Indices of images to use for validation
    VALIDATION_INTERVAL = 25 # Run validation every epoch
    # ---------------------

    # --- WANDB Config ---
    wandb_config = {
        "experiment_num": EX_NUM,
        "backbone": BACKBONE,
        "num_output_channels": NUM_OUTPUT_CHANNELS,
        "pretrained": PRETRAINED,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "initial_lr": INITIAL_LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "min_lr": MIN_LR,
        "bce_weight": BCE_WEIGHT,
        "dice_weight": DICE_WEIGHT,
        "scale_factor": SCALE_FACTOR,
        "crop_h": CROP_H,
        "crop_w": CROP_W,
        "resize_h": RESIZE_H,
        "resize_w": RESIZE_W,
        "validation_interval": VALIDATION_INTERVAL,
        "dataset_mean": DATASET_MEAN, # Log mean/std if used
        "dataset_std": DATASET_STD,
        "optimizer": "AdamW", # Example: Log optimizer type
        "weight_decay": 1e-2, # Example: Log weight decay from optimizer init
        "valid_img_index": VALID_IMG_INDEX,
        "device": DEVICE,
    }
    run_name = f"{EX_NUM}-{BACKBONE}"
    # --- End WANDB Config ---

    print("Setting up datasets and dataloaders...")

    # --- Get all image filenames with annotations ---
    try:
        with open(ANNOTATION_FILE) as f:
            ann_data = json.load(f)
        image_annotations = {item["file_name"]: item["annotations"]
                             for item in ann_data.get("images", [])
                             if isinstance(item, dict) and "file_name" in item and "annotations" in item}
        all_files = os.listdir(IMAGE_DIR)
        all_img_filenames = sorted([fn for fn in all_files if fn.endswith(".tif") and fn in image_annotations])
        if not all_img_filenames:
            raise ValueError(f"No matching .tif files found in {IMAGE_DIR} listed in {ANNOTATION_FILE}")
        print(f"Found {len(all_img_filenames)} total images with annotations.")
    except Exception as e:
        print(f"Error reading image file list or annotations: {e}")
        exit()
    # --- Split filenames ---
    train_img_filenames = [fn for i, fn in enumerate(all_img_filenames) if i not in VALID_IMG_INDEX]
    valid_img_filenames = [fn for i, fn in enumerate(all_img_filenames) if i in VALID_IMG_INDEX]
    print(f"Using {len(train_img_filenames)} images for training and {len(valid_img_filenames)} for validation.")

    # --- Define transformations ---
    # Training transforms (with augmentation)
    transform_train = A.Compose(
        [
            A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomCrop(height=CROP_H, width=CROP_W, p=1.0),
            A.Resize(height=CROP_H, width=CROP_W, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(),
        ]
    )
    # Validation transforms (only resize and tensor conversion)
    transform_valid = A.Compose(
        [
            # Note: RandomCrop is usually not applied to validation data
            A.CenterCrop(height=CROP_H, width=CROP_W, p=1.0),
            ToTensorV2(),
        ]
    )

    print(f"Images will be resized to {RESIZE_H}x{RESIZE_W} for model input.")

    # Save masks for debugging (Consider making this optional)
    debug_output_dir = os.path.join(ROOT_DIR, "outputs", EX_NUM, "check_debug_masks")
    if os.path.exists(debug_output_dir):
        try:
            shutil.rmtree(debug_output_dir, ignore_errors=True)
            os.makedirs(debug_output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error removing existing debug directory: {e}")
    else:
        os.makedirs(debug_output_dir, exist_ok=True)
        print(f"Debug mask directory created: {debug_output_dir}")

    print(f'train idx: {[i for i in range(50) if i not in VALID_IMG_INDEX]}')

    # Ensure FieldSegmentationDataset is correctly implemented and paths/file are valid
    try:
        # --- Initialize WANDB ---
        wandb.init(
            project="solafune-field-segmentation",
            name=run_name,
            config=wandb_config
        )
        # --- End Initialize WANDB ---

        # --- Create Datasets ---
        # Training Dataset
        # We need to pass the filenames directly to the dataset constructor.
        # Since the current FieldSegmentationDataset doesn't support this directly,
        # we modify the approach: Create two separate dataset instances.
        print("Initializing Training Dataset...")
        train_dataset = FieldSegmentationDataset(
            img_dir=IMAGE_DIR,
            ann_json_path=ANNOTATION_FILE,
            cache_dir=CACHE_DIR,
            scale_factor=SCALE_FACTOR,
            transform=transform_train, # Use training transforms
            contact_width=5,
            edge_width=3,
            img_idxes=[i for i in range(50) if i not in VALID_IMG_INDEX],
            mean=DATASET_MEAN,
            std=DATASET_STD,
        )

        # Validation Dataset
        valid_dataset = None
        if valid_img_filenames:
            print("Initializing Validation Dataset...")
            valid_dataset = FieldSegmentationDataset(
                img_dir=IMAGE_DIR,
                ann_json_path=ANNOTATION_FILE,
                cache_dir=CACHE_DIR,
                scale_factor=SCALE_FACTOR,
                transform=transform_valid, # Use validation transforms
                contact_width=5,
                edge_width=3, # Keep consistent
                img_idxes=VALID_IMG_INDEX, # Pass validation indices
                mean=DATASET_MEAN,
                std=DATASET_STD # Use same normalization
            )
            # Manually filter the filenames *after* initialization
            print(f"{valid_dataset.img_filenames}")


        if len(train_dataset) == 0:
            print("Error: Training dataset is empty after filtering. Check file paths and validation indices.")
            exit()
        if valid_dataset is None or len(valid_dataset) == 0:
            print("Warning: Validation dataset is empty or could not be created. Validation will be skipped.")
            valid_dataloader = None # Set dataloader to None
        else:
            print(f"Validation dataset initialized with {len(valid_dataset)} samples.")

        print(f"Training dataset initialized with {len(train_dataset)} samples.")


        # --- Create DataLoaders ---
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True, # Shuffle training data
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE == "cuda" else False,
        )
        # Only create valid_dataloader if valid_dataset exists and is not empty
        if valid_dataset and len(valid_dataset) > 0:
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=BATCH_SIZE, # Can often use a larger batch size for validation
                shuffle=False, # No need to shuffle validation data
                num_workers=NUM_WORKERS,
                pin_memory=True if DEVICE == "cuda" else False,
            )
        else:
             valid_dataloader = None # Ensure it's None if dataset is empty/None

        print("Dataloaders ready.")


        print("Initializing model...")
        # Initialize model with the number of output channels (not classes for BCE loss)
        model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS, img_size=CROP_H)
        model.to(DEVICE)  # Move model to the specified device
        print(f"Model: UNet with {BACKBONE} backbone, {NUM_OUTPUT_CHANNELS} output channels.")

        # Start training
        train_model(
            model=model,
            train_dataloader=train_dataloader, # Pass train dataloader
            valid_dataloader=valid_dataloader, # Pass valid dataloader (can be None)
            validation_interval=VALIDATION_INTERVAL, # Pass validation interval
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            bce_weight=BCE_WEIGHT,
            dice_weight=DICE_WEIGHT,
            initial_lr=INITIAL_LR, # Use config value
            warmup_epochs=WARMUP_EPOCHS, # Use config value
            cosine_decay_epochs=NUM_EPOCHS, # Pass num_epochs, calculation inside train_model
            min_lr=MIN_LR, # Use config value
            # lr_scheduler_t_max and lr_scheduler_eta_min are handled internally now
        )

        # Save the model after training
        model_save_path = os.path.join(ROOT_DIR, "outputs", EX_NUM) # Save in experiment root
        os.makedirs(model_save_path, exist_ok=True)
        final_model_name = "model_final.pth" # Changed from model.path to model.pth
        torch.save(model.state_dict(), os.path.join(model_save_path, final_model_name))
        print(f"Model saved to {os.path.join(model_save_path, final_model_name)}")


    except NameError as e:
        print(f"Error: Class not found (FieldSegmentationDataset or UNet?). Details: {e}")
        print("Ensure 'src' is in PYTHONPATH or run from the project root. Cannot run training.")
    except FileNotFoundError as e:
        print(f"Error: File or directory not found. Please check paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during setup or training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        # --- Finish WANDB Run ---
        if wandb.run: # Check if wandb.init was successful and run is active
            print("Finishing WANDB run...")
            wandb.finish()
        # --- End Finish WANDB Run ---
        print("Training script finished.") # Keep or adjust final message
