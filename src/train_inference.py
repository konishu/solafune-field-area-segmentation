import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import cv2  # Import OpenCV
import os
import math

# Assuming FieldSegmentationDataset is defined in utils.dataset and UNet in models.unet_maxvit
# Adjust imports based on your actual project structure if different
try:
    from utils.dataset import FieldSegmentationDataset  # Corrected class name
    from models.unet_maxvit import UNet
except ImportError:
    print(
        "Warning: Could not import CustomDataset or UNet. Ensure they are defined in the correct paths (src/utils/dataset.py and src/models/unet_maxvit.py)"
    )

    # Define dummy classes if imports fail, to allow the script to load
    class FieldSegmentationDataset:  # Corrected dummy class name
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)  # Return dummy tensors

    class UNet(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.dummy(torch.zeros(x.shape[0], 1))


if __name__ == "__main__":
    # --- Configuration (Should match the training configuration used to save the model) ---
    ROOT_DIR = "/workspace/projects/solafune-field-area-segmentation"
    EX_NUM = "ex1"  # Example experiment number
    IMAGE_DIR = os.path.join(ROOT_DIR, "data/inference_images")  # Path to training images used for inference
    ANNOTATION_FILE = os.path.join(ROOT_DIR, "data/train_annotation.json")  # Needed for dataset initialization
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", EX_NUM, "check")  # Directory where the model is saved
    MODEL_PATH = os.path.join(OUTPUT_DIR, "model.path")  # Path to the saved model state dict
    PREDICTION_DIR = os.path.join(
        OUTPUT_DIR, "train_predictions_inference_script"
    )  # Output directory for predictions from this script
    BACKBONE = "maxvit_small_tf_512.in1k"  # Must match the trained model's backbone
    NUM_OUTPUT_CHANNELS = 3  # Must match the trained model's output channels
    PRETRAINED = False  # Pretrained weights are loaded from MODEL_PATH, not downloaded again
    NUM_WORKERS = 4  # Adjust based on CPU cores
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SCALE_FACTOR = 3  # Must match dataset settings during training
    CROP_H = 512  # Must match dataset settings during training
    CROP_W = 512  # Must match dataset settings during training
    RESIZE_H = 1024  # Must match model input size during training
    RESIZE_W = 1024  # Must match model input size during training
    DATASET_MEAN = None  # Use the same mean/std as during training
    DATASET_STD = None  # Use the same mean/std as during training
    TILE_H = 512  # Tile size for inference (should match CROP_H ideally)
    TILE_W = 512  # Tile size for inference (should match CROP_W ideally)
    STRIDE_H = 128  # Stride for vertical tiling
    STRIDE_W = 128  # Stride for horizontal tiling
    # ---------------------

    print("Setting up dataset for inference...")
    # Define transformations (should be consistent with training, but without random augmentations if desired)
    # Using the same transform as training for simplicity here, but RandomCrop might not be ideal for inference
    # A deterministic crop or resize might be better depending on the goal.
    # Remove RandomCrop for whole image inference
    transform = A.Compose(
        [
            A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_NEAREST),  # Resize to model input size
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
            ann_json_path=ANNOTATION_FILE,
            scale_factor=SCALE_FACTOR,
            transform=transform,
            contact_width=5,  # Match training settings
            edge_width=3,  # Match training settings
            mean=DATASET_MEAN,
            std=DATASET_STD,
        )

        if len(dataset) == 0:
            print(f"Error: Dataset is empty. Check image path '{IMAGE_DIR}' and annotation file '{ANNOTATION_FILE}'.")
            exit()
        print(f"Dataset initialized with {len(dataset)} samples.")

        # Use DataLoader with batch_size=1 and shuffle=False for inference
        # Use batch_size=1 for whole image inference
        inference_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
        print("Dataloader ready for inference (batch_size=1).")

        print("Initializing and loading model...")
        # Initialize model structure
        model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS)

        # Load the saved state dictionary
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            exit()
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
        thresholds = [0.5, 0.1, 0.1]  # Example thresholds for classes 0, 1, and 2

        with torch.no_grad():
            # Iterate through each image in the dataset (batch_size=1)
            progress_bar_infer = tqdm(
                enumerate(inference_dataloader), total=len(inference_dataloader), desc="Inferring Images"
            )
            for idx, batch in progress_bar_infer:
                if batch is None:
                    print(f"Warning: Skipping empty batch at index {idx}")
                    continue

                img_tensor, _ = batch  # Get the single image tensor (C, H, W)
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
                for tile, (y_start, x_start) in tile_progress_bar:
                    tile = tile.unsqueeze(0).to(DEVICE)  # Add batch dim and move to device -> (1, C, H, W)

                    # Perform inference on the tile
                    tile_output = model(tile)  # Output shape (1, NUM_OUTPUT_CHANNELS, TILE_H, TILE_W)
                    tile_output = torch.sigmoid(tile_output).squeeze(
                        0
                    )  # Remove batch dim -> (NUM_OUTPUT_CHANNELS, TILE_H, TILE_W)

                    # Determine the region in the full map corresponding to this tile
                    y_end = min(y_start + TILE_H, original_h)
                    x_end = min(x_start + TILE_W, original_w)
                    h_tile, w_tile = (
                        y_end - y_start,
                        x_end - x_start,
                    )  # Actual size of the tile region in the original image

                    # Add the prediction to the full map and update the count map
                    # Ensure we only take the relevant part of the tile_output if padding occurred
                    full_prediction_map[:, y_start:y_end, x_start:x_end] += tile_output[:, :h_tile, :w_tile]
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

                    # Save each class mask separately (optional, can be removed if only combined is needed)
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
                        if final_mask_np.shape[2] > 2:
                            combined_mask_bgr[:, :, 2] = (final_mask_np[:, :, 2] * 255).astype(np.uint8)  # Red

                        combined_output_path = os.path.join(
                            PREDICTION_DIR, f"{os.path.splitext(output_filename_base)[0]}_combined.png"
                        )
                        cv2.imwrite(combined_output_path, combined_mask_bgr)
                        print(f"Output mask saved for {original_img_filename} to {combined_output_path}")
                        progress_bar_infer.set_postfix(saved=original_img_filename)

                    except Exception as e:
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
    except Exception as e:
        print(f"An unexpected error occurred during setup or inference: {e}")
