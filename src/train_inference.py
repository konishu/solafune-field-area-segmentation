import argparse
import os
import shutil
import math
import os
import sys

import albumentations as A
import cv2  # Import OpenCV
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.unet_maxvit import UNet
from utils.dataset import FieldSegmentationDataset  # Corrected class name


# --- Helper Functions (Assuming these are defined elsewhere or copied from train.py/utils) ---
def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit()
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit()


# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Inference script for field segmentation")
parser.add_argument(
    "--config",
    type=str,
    default="../configs/ex9.yaml",  # Default config path
    help="Path to the configuration YAML file",
)
args = parser.parse_args()

# --- Load Configuration ---
cfg = load_config(args.config)
print("Configuration loaded:")


# --- Configuration (Derived from loaded cfg) ---
ROOT_DIR = cfg["experiment"]["root_dir"]
EX_NUM = cfg["experiment"]["ex_num"]
OUTPUT_DIR_BASE = cfg["experiment"]["output_dir_base"]
CACHE_DIR_BASE = cfg["experiment"]["cache_dir_base"]
# Construct full paths relative to ROOT_DIR
OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_BASE, EX_NUM)  # Keep check subdir for debug outputs if needed
CACHE_DIR = os.path.join(ROOT_DIR, CACHE_DIR_BASE, EX_NUM, "cache")
IMAGE_DIR = os.path.join(ROOT_DIR, cfg["test"]["image_dir"])
ANNOTATION_FILE = os.path.join(ROOT_DIR, cfg["data"]["annotation_file"])
MODEL_PATH = os.path.join(ROOT_DIR, cfg["test"]["model_dir"], "model_best_dice.pth")  # Use saved model name from config
PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "train_predictions_inference_script",  # Keep the specific output folder name for now
)
BACKBONE = cfg["model"]["backbone"]
NUM_OUTPUT_CHANNELS = cfg["model"]["num_output_channels"]
PRETRAINED = False  # Set to False for inference as we load weights from MODEL_PATH
NUM_WORKERS = cfg["validation"]["num_workers"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Dataset/Preprocessing parameters should match training
SCALE_FACTOR = cfg["data"]["scale_factor"]
CONTACT_WIDTH = cfg["data"]["contact_width"]
EDGE_WIDTH = cfg["data"]["edge_width"]
# CROP_H/W might correspond to the size used during validation transforms if different from resize
CROP_H = cfg["training"]["crop_h"]  # Assuming validation used cropping
CROP_W = cfg["training"]["crop_w"]
# RESIZE_H/W is the actual model input size
RESIZE_H = cfg["training"]["resize_h"]
RESIZE_W = cfg["training"]["resize_w"]
DATASET_MEAN = cfg["data"].get("mean", None)  # Use get for optional keys
DATASET_STD = cfg["data"].get("std", None)
TILE_H = cfg["validation"].get("tile_h", 512)  # Use crop size if tile size not specified
TILE_W = cfg["validation"].get("tile_w", 512)  # Use crop size if tile size not specified
STRIDE_H = cfg["validation"].get("stride_h", 256)  # Default stride
STRIDE_W = cfg["validation"].get("stride_w", 256)  # Default stride


if __name__ == "__main__":
    print("Setting up dataset for inference...")
    # Define transformations (should be consistent with training, but without random augmentations if desired)
    # Using the same transform as training for simplicity here, but RandomCrop might not be ideal for inference
    # A deterministic crop or resize might be better depending on the goal.
    # Remove RandomCrop for whole image inference
    # Remove Resize from transform, apply it per tile later
    transform = A.Compose(
        [
            # A.Resize(...) # Removed: Resize will be done per tile
            # A.PadIfNeeded(...) # Add padding if used during training
            ToTensorV2(),
        ]
    )

    # Ensure prediction directory exists
    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
        print(f"Prediction directory created: {PREDICTION_DIR}")

    try:
        # Initialize dataset
        dataset = FieldSegmentationDataset(
            img_dir=IMAGE_DIR,
            scale_factor=SCALE_FACTOR,
            transform=transform,
            contact_width=CONTACT_WIDTH,  # Match training settings
            edge_width=EDGE_WIDTH,  # Match training settings
            mean=DATASET_MEAN,
            std=DATASET_STD,
            cache_dir=CACHE_DIR,
            is_test_mode=True,  # Set to True for inference
        )

        if len(dataset) == 0:
            print(f"Error: Dataset is empty. Check image path '{IMAGE_DIR}' and annotation file '{ANNOTATION_FILE}'.")
            sys.exit()
        print(f"Dataset initialized with {len(dataset)} samples.")

        # Use DataLoader with batch_size=1 and shuffle=False for inference
        # Use batch_size=1 for whole image inference
        inference_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
        print("Dataloader ready for inference (batch_size=1).")

        print("Initializing and loading model...")
        # Initialize model structure
        model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS, img_size=RESIZE_W)

        # Load the saved state dictionary
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            sys.exit()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print(f"Model loaded from {MODEL_PATH} and moved to {DEVICE}.")

        # --- Tiling Inference Helper Function ---
        def create_tiles(image_tensor, tile_h, tile_w, stride_h, stride_w):
            """Creates overlapping tiles from an image tensor (C, H, W)."""
            c, h, w = image_tensor.shape
            tiles = []
            coords = []  # Store (y, x) top-left coordinates of each tile

            # Calculate number of tiles needed in each dimension
            num_tiles_h = math.ceil((h - tile_h) / stride_h) + 1 if h > tile_h else 1
            num_tiles_w = math.ceil((w - tile_w) / stride_w) + 1 if w > tile_w else 1

            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    y_start = i * stride_h
                    x_start = j * stride_w

                    # Ensure tile does not go out of bounds (adjust end coordinates)
                    y_end = min(y_start + tile_h, h)
                    x_end = min(x_start + tile_w, w)

                    # Adjust start coordinates if tile is smaller than tile_h/tile_w at the edge
                    y_start = max(0, y_end - tile_h)
                    x_start = max(0, x_end - tile_w)

                    tile = image_tensor[:, y_start:y_end, x_start:x_end]

                    # Pad if tile is smaller than target size (important for model input)
                    pad_h = tile_h - tile.shape[1]
                    pad_w = tile_w - tile.shape[2]
                    if pad_h > 0 or pad_w > 0:
                        # Pad tuple format: (pad_left, pad_right, pad_top, pad_bottom)
                        padding = (0, pad_w, 0, pad_h)
                        tile = torch.nn.functional.pad(
                            tile, padding, "reflect"
                        )  # Use reflect padding or adjust as needed

                    tiles.append(tile)
                    coords.append((y_start, x_start))  # Store original top-left corner before potential edge adjustment

            return tiles, coords

        # --- Inference with Tiling ---
        print("\nStarting Tiling Inference...")
        model.eval()  # Set model to evaluation mode
        thresholds = [0.3, 0.1, 0.1]  # Example thresholds for classes 0, 1, and 2

        with torch.no_grad():
            # Iterate through each image in the dataset (batch_size=1)
            progress_bar_infer = tqdm(
                enumerate(inference_dataloader), total=len(inference_dataloader), desc="Inferring Images"
            )
            for idx, batch in progress_bar_infer:
                if batch is None:
                    print(f"Warning: Skipping empty batch at index {idx}")
                    continue

                img_tensor, _, _ = batch  # Get the single image tensor (C, H, W)
                img_tensor = img_tensor.squeeze(0)  # Remove batch dimension -> (C, H, W)
                c, original_h, original_w = img_tensor.shape  # Get original dimensions *after* transform (Resize)

                # Initialize full prediction map and count map on the correct device
                full_prediction_map = torch.zeros(
                    (NUM_OUTPUT_CHANNELS, original_h, original_w), dtype=torch.float32, device=DEVICE
                )
                count_map = torch.zeros((original_h, original_w), dtype=torch.float32, device=DEVICE)

                # Create tiles from the image
                tiles, coords = create_tiles(img_tensor, TILE_H, TILE_W, STRIDE_H, STRIDE_W)
                print(f"Image {idx}: Created {len(tiles)} tiles.")

                # --- Process tiles ---
                tile_progress_bar = tqdm(zip(tiles, coords), total=len(tiles), desc=f"  Tiles Img {idx}", leave=False)
                for tile_data, (y_start, x_start) in tile_progress_bar:
                    tile_tensor = tile_data.to(DEVICE)  # Tile is already (C, TILE_H, TILE_W)

                    # --- Resize tile to model input size ---
                    # Add batch dimension for interpolate and model input
                    tile_tensor_batch = tile_tensor.unsqueeze(0)  # (1, C, TILE_H, TILE_W)
                    resized_tile_tensor = torch.nn.functional.interpolate(
                        tile_tensor_batch, size=(RESIZE_H, RESIZE_W), mode="bilinear", align_corners=False
                    )  # (1, C, RESIZE_H, RESIZE_W)

                    # --- Perform inference on the resized tile ---
                    tile_output = model(
                        resized_tile_tensor
                    )  # Output shape (1, NUM_OUTPUT_CHANNELS, RESIZE_H, RESIZE_W)
                    tile_output = torch.sigmoid(tile_output)

                    # --- Resize prediction back to original tile size ---
                    resized_back_output = torch.nn.functional.interpolate(
                        tile_output, size=(TILE_H, TILE_W), mode="bilinear", align_corners=False
                    ).squeeze(0)  # Remove batch dim -> (NUM_OUTPUT_CHANNELS, TILE_H, TILE_W)

                    # Determine the region in the full map corresponding to this tile
                    y_end = min(y_start + TILE_H, original_h)
                    x_end = min(x_start + TILE_W, original_w)
                    h_tile, w_tile = (
                        y_end - y_start,
                        x_end - x_start,
                    )  # Actual size of the tile region in the original image

                    # Add the resized-back prediction to the full map and update the count map
                    # Ensure we only take the relevant part of the resized_back_output
                    full_prediction_map[:, y_start:y_end, x_start:x_end] += resized_back_output[:, :h_tile, :w_tile]
                    count_map[y_start:y_end, x_start:x_end] += 1

                # --- Average predictions ---
                # Avoid division by zero
                count_map[count_map == 0] = 1e-6  # Replace 0s with a small number
                averaged_prediction_map = full_prediction_map / count_map.unsqueeze(
                    0
                )  # Add channel dim to count_map for broadcasting

                # --- Apply Thresholds ---
                final_mask = torch.zeros_like(averaged_prediction_map, dtype=torch.uint8)
                for class_idx, threshold in enumerate(thresholds):
                    final_mask[class_idx, :, :] = (
                        averaged_prediction_map[class_idx, :, :] > threshold
                    ).byte()  # Use byte for 0/1

                # --- Save the final mask ---
                if idx < len(dataset.img_filenames):
                    original_img_filename = dataset.img_filenames[idx]
                    output_filename_base = os.path.splitext(original_img_filename)[0] + "_pred_tiled.png"  # Add suffix

                    # Convert final mask tensor to numpy array (H, W, C) for saving
                    final_mask_np = final_mask.cpu().numpy()  # (C, H, W)
                    final_mask_np = np.transpose(final_mask_np, (1, 2, 0))  # (H, W, C)

                    # --- Verify output mask size against input image size ---
                    output_h, output_w = final_mask_np.shape[:2]
                    if output_h != original_h or output_w != original_w:
                        print(f"Warning: Size mismatch for {original_img_filename}!")
                        print(f"  Input size (after scale_factor): ({original_h}, {original_w})")
                        print(f"  Output mask size: ({output_h}, {output_w})")
                    else:
                        print(
                            f"Output mask size ({output_h}, {output_w}) matches input size for {original_img_filename}."
                        )
                    # ---------------------------------------------------------

                    # Save each class mask separately
                    try:
                        for class_idx in range(final_mask_np.shape[2]):
                            class_output_path = os.path.join(
                                PREDICTION_DIR, f"{os.path.splitext(output_filename_base)[0]}_class_{class_idx}.png"
                            )
                            mask_to_save = (final_mask_np[:, :, class_idx] * 255).astype(np.uint8)
                            cv2.imwrite(class_output_path, mask_to_save)

                        # Save the combined 3-channel mask (BGR order for cv2.imwrite)
                        # Class 0 -> Blue, Class 1 -> Green, Class 2 -> Red
                        combined_mask_bgr = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                        if final_mask_np.shape[2] > 0:
                            combined_mask_bgr[:, :, 0] = (final_mask_np[:, :, 0] * 255).astype(np.uint8)  # Blue
                        if final_mask_np.shape[2] > 1:
                            combined_mask_bgr[:, :, 1] = (final_mask_np[:, :, 1] * 255).astype(np.uint8)  # Green
                        if final_mask_np.shape[2] > 2:  # noqa: PLR2004
                            combined_mask_bgr[:, :, 2] = (final_mask_np[:, :, 2] * 255).astype(np.uint8)  # Red

                        combined_output_path = os.path.join(
                            PREDICTION_DIR, f"{os.path.splitext(output_filename_base)[0]}_combined.png"
                        )
                        cv2.imwrite(combined_output_path, combined_mask_bgr)
                        print(f"Output mask saved for {original_img_filename} to {combined_output_path}")
                        progress_bar_infer.set_postfix(saved=original_img_filename)

                    except (OSError, TypeError, cv2.error) as e:  # Catch more specific errors
                        print(f"Error saving prediction for {original_img_filename}: {e}")
                else:
                    print(f"Warning: Index {idx} out of bounds for dataset filenames.")

        print(f"\nTiling Inference complete. Predictions saved to {PREDICTION_DIR}")

    except NameError:
        print(
            "Error: FieldSegmentationDataset or UNet class not found. Ensure 'src' is in PYTHONPATH or run from the project root."
        )
    except FileNotFoundError as e:
        print(f"Error: File or directory not found. Please check paths. Details: {e}")
    except (RuntimeError, ValueError, ImportError, TypeError) as e:  # Catch more specific setup/inference errors
        print(f"An error occurred during setup or inference: {e}")
