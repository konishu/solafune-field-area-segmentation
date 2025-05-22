import os
import shutil
import json  # アノテーションファイル読み込みのため追加
import argparse  # YAML読み込みのため追加
import yaml  # YAML読み込みのため追加
import numpy as np
import torch
import torchvision  # For image logging
import wandb  # Import WANDB
from dotenv import load_dotenv  # Import dotenv

import albumentations as A
import cv2  # Import OpenCV
from albumentations.pytorch import ToTensorV2
from models.unet_maxvit import UNet
from torch import nn, optim
from torch.optim import lr_scheduler  # Import LR scheduler (LinearLR, CosineAnnealingLR, SequentialLR)
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
    train_dataloader,  # Renamed from dataloader
    valid_dataloader,  # Added for validation
    validation_interval,  # Added for validation frequency
    num_epochs=500,
    device="cuda",
    bce_weight=0.5,
    dice_weight=0.5,
    initial_lr=4e-4,  # Add initial_lr parameter
    warmup_epochs=10,  # Add warmup_epochs parameter
    cosine_decay_epochs=500,  # Note: This is effectively num_epochs in the scheduler logic below
    min_lr=1e-6,
    # lr_scheduler_t_max and lr_scheduler_eta_min are effectively num_epochs and min_lr for CosineAnnealingLR
    weight_decay=1e-2,  # Default value if not passed from main
    wandb_log_images=True,  # Default value if not passed from main
    wandb_num_images_to_log=4,  # Default value if not passed from main
    accumulation_steps=1,  # <<< Minimal change: Added argument with default
    pos_weight_ratio=[0.11, 99.0, 19.0],  # Default pos_weight for BCE loss
):
    """
    Trains the U-Net model using BCE + Dice loss with Linear Warmup + Cosine Decay LR schedule,
    and performs validation periodically.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, switching to CPU.")
        device = "cpu"
    model.to(device)

    # Use BCEWithLogitsLoss (reduction='mean' is simpler here)
    pos_weight = torch.tensor(
        pos_weight_ratio,
        device=device,
        dtype=torch.float,
    ).view(1, -1, 1, 1)
    criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion_bce = nn.BCEWithLogitsLoss()  # Default reduction='mean'
    # No need for separate Dice criterion instance if using the function directly

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    actual_cosine_epochs = num_epochs - warmup_epochs
    if actual_cosine_epochs <= 0:
        print(f"Warning: warmup_epochs ({warmup_epochs}) >= num_epochs ({num_epochs}). Cosine decay part will not run.")
        actual_cosine_epochs = 1

    if warmup_epochs >= num_epochs:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_epochs)
    elif warmup_epochs > 0:
        scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_cosine_epochs, eta_min=min_lr)
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    print(
        f"Starting training on {device} for {num_epochs} epochs (BCE weight: {bce_weight}, Dice weight: {dice_weight}, Initial LR: {initial_lr}, Weight Decay: {weight_decay}, Accumulation Steps: {accumulation_steps})..."
    )
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0

        # <<< Minimal change: Use enumerate for batch_idx
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            leave=False,
        )

        # <<< Minimal change: optimizer.zero_grad() moved to after accumulation step
        # optimizer.zero_grad() # Removed from here

        for batch_idx, batch in progress_bar:
            if batch is None:
                print("Warning: Skipping empty batch.")
                continue

            imgs, masks, filenames = batch
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.float) / 255.0

            # optimizer.zero_grad() # <<< Minimal change: Removed from here

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(imgs)
                # print(
                #     f"Debug: outputs.shape = {outputs.shape}, masks.shape = {masks.shape}"
                # )  # <<< デバッグ用プリント追加
                if outputs.shape[2:] != masks.shape[2:]:
                    raise ValueError(f"Spatial dimension mismatch! Output: {outputs.shape}, Mask: {masks.shape}")
                if outputs.shape[1] != masks.shape[1]:
                    raise ValueError(f"Channel dimension mismatch! Output: {outputs.shape}, Mask: {masks.shape}")

                loss_bce = criterion_bce(outputs, masks.float())
                loss_dice = dice_loss(outputs, masks)
                total_loss = bce_weight * loss_bce + dice_weight * loss_dice

            # <<< Minimal change: Normalize loss for accumulation
            if accumulation_steps > 1:
                loss_to_backward = total_loss / accumulation_steps
            else:
                loss_to_backward = total_loss

            scaler.scale(loss_to_backward).backward()

            # <<< Minimal change: Step optimizer and zero_grad only after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero gradients after optimizer step

            if torch.isnan(total_loss) or torch.isnan(loss_bce) or torch.isnan(loss_dice):
                print(f"NaN loss encountered in epoch {epoch + 1}, batch {batch_idx}.")  # Added batch_idx
                print(f"{filenames=}")
                print(
                    f"  Loss values: total={total_loss.item():.4f}, bce={loss_bce.item():.4f}, dice={loss_dice.item():.4f}"
                )
                print("  --- Debugging Tensor Stats ---")
                print(f"  Outputs (model predictions) stats:")
                print(f"    Shape: {outputs.shape}")
                print(f"    Contains NaN: {torch.isnan(outputs).any().item()}")
                print(f"    Contains Inf: {torch.isinf(outputs).any().item()}")
                if not torch.isnan(outputs).any() and not torch.isinf(outputs).any():
                    print(
                        f"    Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}, Mean: {outputs.mean().item():.4f}, Std: {outputs.std().item():.4f}"
                    )
                print(f"  Masks (target) stats:")
                print(f"    Shape: {masks.shape}")
                print(f"    Contains NaN: {torch.isnan(masks).any().item()}")
                print(f"    Contains Inf: {torch.isinf(masks).any().item()}")
                if not torch.isnan(masks).any() and not torch.isinf(masks).any():
                    print(
                        f"    Min: {masks.min().item():.4f}, Max: {masks.max().item():.4f}, Mean: {masks.mean().item():.4f}, Std: {masks.std().item():.4f}"
                    )
                print("  -----------------------------")
                # If NaN, and it's an accumulation step, still zero grad to prevent carry-over
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)  # Use set_to_none for potential minor memory saving
                continue

            running_loss += total_loss.item()  # Log the original, un-normalized loss for this physical batch
            running_bce_loss += loss_bce.item()
            running_dice_loss += loss_dice.item()
            progress_bar.set_postfix(loss=total_loss.item(), bce=loss_bce.item(), dice=loss_dice.item())

        # <<< Minimal change: Handle final optimizer step if dataloader size is not a multiple of accumulation_steps
        if len(train_dataloader) > 0 and (len(train_dataloader) % accumulation_steps != 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_bce_loss = running_bce_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_dice_loss = running_dice_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} Avg Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce_loss:.4f}, Dice: {avg_train_dice_loss:.4f}) LR: {current_lr:.6f}"
        )

        if wandb.run:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_train_bce_loss": avg_train_bce_loss,
                    "avg_train_dice_loss": avg_train_dice_loss,
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )

        scheduler.step()

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            running_val_loss = 0.0
            running_val_bce_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_dice_coeff = 0.0
            num_val_samples = 0
            if valid_dataloader and len(valid_dataloader) > 0:
                val_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]", leave=False)
                with torch.no_grad():
                    for i_val_batch, val_batch in enumerate(val_progress_bar):
                        if val_batch is None:
                            continue
                        val_imgs, val_masks, _ = val_batch
                        val_imgs = val_imgs.to(device)
                        val_masks = val_masks.to(device, dtype=torch.float) / 255.0
                        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                            val_outputs = model(val_imgs)
                            if val_outputs.shape[2:] != val_masks.shape[2:]:
                                print(
                                    f"Warning: Validation shape mismatch! Output: {val_outputs.shape}, Mask: {val_masks.shape}"
                                )
                                continue
                            if val_outputs.shape[1] != val_masks.shape[1]:
                                print(
                                    f"Warning: Validation channel mismatch! Output: {val_outputs.shape}, Mask: {val_masks.shape}"
                                )
                                continue
                            val_loss_bce = criterion_bce(val_outputs, val_masks)
                            val_loss_dice = dice_loss(val_outputs, val_masks)
                            val_total_loss = bce_weight * val_loss_bce + dice_weight * val_loss_dice
                        if not torch.isnan(val_total_loss):
                            running_val_loss += val_total_loss.item()
                            running_val_bce_loss += val_loss_bce.item()
                            running_val_dice_loss += val_loss_dice.item()
                            val_progress_bar.set_postfix(
                                loss=val_total_loss.item(), bce=val_loss_bce.item(), dice=val_loss_dice.item()
                            )
                        else:
                            print(f"Warning: NaN validation loss encountered in epoch {epoch + 1}.")
                            continue
                        with torch.no_grad():
                            val_outputs_sigmoid = torch.sigmoid(val_outputs)
                            dice_coeffs_batch = dice_coeff(val_outputs_sigmoid, val_masks, smooth=1.0, epsilon=1e-6)
                            dice_per_item = dice_coeffs_batch.mean(dim=1)
                            running_val_dice_coeff += dice_per_item.sum().item()
                            num_val_samples += val_imgs.size(0)
                        if wandb.run and wandb_log_images and i_val_batch == 0:
                            log_images = []
                            num_samples_to_log = min(wandb_num_images_to_log, val_imgs.size(0))
                            val_imgs_cpu = val_imgs[:num_samples_to_log].cpu().detach()
                            val_masks_cpu = val_masks[:num_samples_to_log].cpu().detach()
                            val_outputs_cpu = val_outputs[:num_samples_to_log].cpu().detach()
                            preds_sigmoid = torch.sigmoid(val_outputs_cpu)
                            preds_binary = (preds_sigmoid > 0.5).float()
                            for i in range(num_samples_to_log):
                                img = val_imgs_cpu[i][[2, 3, 4], ...]
                                mask_gt_combined = val_masks_cpu[i]
                                mask_pred_combined = preds_binary[i]
                                img = torch.clamp(img, 0, 1)
                                mask_gt_combined = torch.clamp(mask_gt_combined, 0, 1)
                                mask_pred_combined = torch.clamp(mask_pred_combined, 0, 1)
                                combined_image = torchvision.utils.make_grid(
                                    [img, mask_gt_combined, mask_pred_combined], nrow=3, padding=2, normalize=False
                                )
                                log_images.append(
                                    wandb.Image(
                                        combined_image, caption=f"Epoch {epoch + 1} Sample {i} (Img | GT | Pred)"
                                    )
                                )
                            if log_images:
                                wandb.log({"validation_samples": log_images}, step=epoch + 1)
                avg_val_loss = running_val_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_bce_loss = running_val_bce_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_dice_loss = running_val_dice_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_dice = running_val_dice_coeff / num_val_samples if num_val_samples > 0 else 0.0
                print(
                    f"Epoch {epoch + 1} Avg Valid Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce_loss:.4f}, Dice: {avg_val_dice_loss:.4f}) Dice Coeff: {avg_val_dice:.4f}"
                )
                if wandb.run:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "avg_val_loss": avg_val_loss,
                            "avg_val_bce_loss": avg_val_bce_loss,
                            "avg_val_dice_loss": avg_val_dice_loss,
                            "avg_val_dice_coeff": avg_val_dice,
                        },
                        step=epoch + 1,
                    )
            else:
                print(f"Epoch {epoch + 1} Validation skipped (empty or no dataloader).")
            model.train()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net model for field segmentation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"Configuration loaded from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"Error loading or parsing configuration file: {e}")
        exit()

    load_dotenv()

    ROOT_DIR = cfg["experiment"]["root_dir"]
    EX_NUM = cfg["experiment"]["ex_num"]
    OUTPUT_DIR_BASE = cfg["experiment"]["output_dir_base"]
    CACHE_DIR_BASE = cfg["experiment"]["cache_dir_base"]
    OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_BASE, EX_NUM)
    CACHE_DIR = os.path.join(ROOT_DIR, CACHE_DIR_BASE, EX_NUM, "cache")
    IMAGE_DIR = os.path.join(ROOT_DIR, cfg["data"]["image_dir"])
    ANNOTATION_FILE = os.path.join(ROOT_DIR, cfg["data"]["annotation_file"])
    POS_WEIGHT_RATIO = cfg["data"]["pos_weight_ratio"]

    VALID_IMG_INDEX = cfg["data"]["valid_img_index"]
    NUM_WORKERS = cfg["data"]["num_workers"]
    SCALE_FACTOR = cfg["data"]["scale_factor"]
    CONTACT_WIDTH = cfg["data"]["contact_width"]
    EDGE_WIDTH = cfg["data"]["edge_width"]
    DATASET_MEAN = cfg["data"]["dataset_mean"]
    DATASET_STD = cfg["data"]["dataset_std"]

    BACKBONE = cfg["model"]["backbone"]
    NUM_OUTPUT_CHANNELS = cfg["model"]["num_output_channels"]
    PRETRAINED = cfg["model"]["pretrained"]

    BATCH_SIZE = cfg["training"]["batch_size"]
    NUM_EPOCHS = cfg["training"]["num_epochs"]
    DEVICE = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    if cfg["training"]["device"] == "cuda" and DEVICE == "cpu":
        print("Warning: CUDA requested in config but not available. Using CPU.")

    CROP_H = cfg["training"]["crop_h"]
    CROP_W = cfg["training"]["crop_w"]
    RESIZE_H = cfg["training"]["resize_h"]
    RESIZE_W = cfg["training"]["resize_w"]
    BCE_WEIGHT = cfg["training"]["bce_weight"]
    DICE_WEIGHT = cfg["training"]["dice_weight"]
    INITIAL_LR = cfg["training"]["initial_lr"]
    WARMUP_EPOCHS = cfg["training"]["warmup_epochs"]
    MIN_LR = cfg["training"]["min_lr"]
    WEIGHT_DECAY = cfg["training"]["weight_decay"]
    VALIDATION_INTERVAL = cfg["training"]["validation_interval"]
    # <<< Minimal change: Read accumulation_steps from config
    ACCUMULATION_STEPS = cfg["training"].get("accumulation_steps", 1)

    WANDB_PROJECT = cfg["wandb"]["project"]
    WANDB_LOG_IMAGES = cfg["wandb"]["log_images"]
    WANDB_LOG_IMAGE_FREQ = cfg["wandb"].get("log_image_freq", VALIDATION_INTERVAL)
    WANDB_NUM_IMAGES_TO_LOG = cfg["wandb"].get("num_images_to_log", 4)
    run_name = f"{EX_NUM}-{BACKBONE}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Output directory (check): {OUTPUT_DIR}")
    print(f"Cache directory: {CACHE_DIR}")

    print("Setting up datasets and dataloaders...")

    try:
        with open(ANNOTATION_FILE) as f:
            ann_data = json.load(f)
        image_annotations = {
            item["file_name"]: item["annotations"]
            for item in ann_data.get("images", [])
            if isinstance(item, dict) and "file_name" in item and "annotations" in item
        }
        all_files = os.listdir(IMAGE_DIR)
        all_img_filenames = sorted([fn for fn in all_files if fn.endswith(".tif") and fn in image_annotations])
        if not all_img_filenames:
            raise ValueError(f"No matching .tif files found in {IMAGE_DIR} listed in {ANNOTATION_FILE}")
        print(f"Found {len(all_img_filenames)} total images with annotations.")
    except Exception as e:
        print(f"Error reading image file list or annotations: {e}")
        exit()
    train_img_filenames = [fn for i, fn in enumerate(all_img_filenames) if i not in VALID_IMG_INDEX]
    valid_img_filenames = [fn for i, fn in enumerate(all_img_filenames) if i in VALID_IMG_INDEX]
    print(f"Using {len(train_img_filenames)} images for training and {len(valid_img_filenames)} for validation.")

    transform_train = A.Compose(
        [
            A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomCrop(height=CROP_H, width=CROP_W, p=1.0),
            A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(),
        ]
    )
    transform_valid = A.Compose(
        [
            A.CenterCrop(height=CROP_H, width=CROP_W, p=1.0),
            A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(),
        ]
    )

    print(f"Images will be cropped to {CROP_H}x{CROP_W} then resized to {RESIZE_H}x{RESIZE_W} for model input.")
    train_idxes = [i for i in range(len(all_img_filenames)) if i not in VALID_IMG_INDEX]
    print(f"train idx: {train_idxes}")

    try:
        wandb_config_log = {
            "config_file": args.config,
            "experiment": cfg["experiment"],
            "data": cfg["data"],
            "model": cfg["model"],
            "training": cfg["training"],
            "optimizer": "AdamW",
        }
        wandb.init(project=WANDB_PROJECT, name=run_name, config=wandb_config_log)

        print("Initializing Training Dataset...")
        train_dataset = FieldSegmentationDataset(
            img_dir=IMAGE_DIR,
            ann_json_path=ANNOTATION_FILE,
            cache_dir=CACHE_DIR,
            scale_factor=SCALE_FACTOR,
            transform=transform_train,
            contact_width=CONTACT_WIDTH,
            edge_width=EDGE_WIDTH,
            img_idxes=train_idxes,
            mean=DATASET_MEAN,
            std=DATASET_STD,
        )

        valid_dataset = None
        if valid_img_filenames:
            print("Initializing Validation Dataset...")
            valid_dataset = FieldSegmentationDataset(
                img_dir=IMAGE_DIR,
                ann_json_path=ANNOTATION_FILE,
                cache_dir=CACHE_DIR,
                scale_factor=SCALE_FACTOR,
                transform=transform_valid,
                contact_width=CONTACT_WIDTH,
                edge_width=EDGE_WIDTH,
                img_idxes=VALID_IMG_INDEX,
                mean=DATASET_MEAN,
                std=DATASET_STD,
            )
            
        print(f'{train_idxes=}')
        print(f'{VALID_IMG_INDEX=}')
        

        if len(train_dataset) == 0:
            print("Error: Training dataset is empty after filtering. Check file paths and validation indices.")
            exit()
        if valid_dataset is None or len(valid_dataset) == 0:
            print("Warning: Validation dataset is empty or could not be created. Validation will be skipped.")
            valid_dataloader = None
        else:
            print(f"Validation dataset initialized with {len(valid_dataset)} samples.")
        print(f"Training dataset initialized with {len(train_dataset)} samples.")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE == "cuda" else False,
            drop_last=True,
        )
        if valid_dataset and len(valid_dataset) > 0:
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True if DEVICE == "cuda" else False,
                drop_last=False,
            )
        else:
            valid_dataloader = None

        print("Dataloaders ready.")
        print("Initializing model...")
        model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS, img_size=RESIZE_H)
        model.to(DEVICE)
        print(
            f"Model: UNet with {BACKBONE} backbone, {NUM_OUTPUT_CHANNELS} output channels, input size {RESIZE_H}x{RESIZE_W}."
        )

        train_model(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            validation_interval=VALIDATION_INTERVAL,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            bce_weight=BCE_WEIGHT,
            dice_weight=DICE_WEIGHT,
            initial_lr=INITIAL_LR,
            warmup_epochs=WARMUP_EPOCHS,
            cosine_decay_epochs=NUM_EPOCHS,
            min_lr=MIN_LR,
            weight_decay=WEIGHT_DECAY,
            wandb_log_images=WANDB_LOG_IMAGES,
            wandb_num_images_to_log=WANDB_NUM_IMAGES_TO_LOG,
            accumulation_steps=ACCUMULATION_STEPS,
            pos_weight_ratio=POS_WEIGHT_RATIO,
        )

        model_save_path = os.path.join(ROOT_DIR, OUTPUT_DIR_BASE, EX_NUM)
        os.makedirs(model_save_path, exist_ok=True)
        final_model_name = "model_final.pth"
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

        traceback.print_exc()
    finally:
        if wandb.run:
            print("Finishing WANDB run...")
            wandb.finish()
        print("Training script finished.")
