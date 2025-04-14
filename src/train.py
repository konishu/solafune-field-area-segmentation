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
import shutil
import rasterio
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
    # Use BCEWithLogitsLoss with reduction='none' to calculate per-element loss
    criterion = nn.BCEWithLogitsLoss(reduction='none')
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

            # # バッチの形状をログに出力
            # print(f"Batch shapes - imgs: {imgs.shape}, masks: {masks.shape}")

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
            
            # # モデルの出力形状とマスクの形状を損失計算直前に再度確認
            # print(f"[Before Loss] Model output shape: {outputs.shape}, dtype: {outputs.dtype}")
            # print(f"[Before Loss] Masks shape: {masks.shape}, dtype: {masks.dtype}")

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

            # Calculate per-element loss
            pixel_losses = criterion(outputs, masks) # Shape: (N, C, H, W)

            # Calculate mean loss per channel (class)
            loss_field = pixel_losses[:, 0, :, :].mean()
            loss_edge = pixel_losses[:, 1, :, :].mean()
            loss_contact = pixel_losses[:, 2, :, :].mean()

            # Calculate the overall mean loss for backpropagation
            total_loss = pixel_losses.mean()

            # Backward pass and optimize using the total loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item(), field=loss_field.item(), edge=loss_edge.item(), contact=loss_contact.item())

        avg_loss = running_loss / len(dataloader)
        # Calculate average losses for the epoch (optional, for cleaner logging)
        # Note: This requires storing per-class losses per batch and averaging at the end
        # For simplicity, the progress bar already shows the last batch's per-class loss.
        # We print the average total loss here.
        print(f"Epoch {epoch+1} Average Total Loss: {avg_loss:.4f}")
        # You could add more detailed logging here if needed, e.g., average per-class loss for the epoch

    print("Training finished.")


if __name__ == "__main__":
    # --- Configuration ---
    ROOT_DIR = '/workspace/projects/solafune-field-area-segmentation'
    EX_NUM = 'ex0' # Example experiment number
    IMAGE_DIR = os.path.join(ROOT_DIR, 'data/train_images') # Path to training images (adjust if needed)
    ANNOTATION_FILE = os.path.join(ROOT_DIR, 'data/train_annotation.json') # Path to training annotations (adjust if needed)
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs',EX_NUM,'check') # Path to save model outputs
    BACKBONE = 'maxvit_small_tf_512.in1k' # Example backbone
    NUM_OUTPUT_CHANNELS = 3 # Number of output channels (field, edge, contact)
    PRETRAINED = True
    BATCH_SIZE = 1 # Adjust based on GPU memory
    NUM_WORKERS = 4 # Adjust based on CPU cores
    NUM_EPOCHS = 1 # Number of training epochs
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_H = 512 # Example, not directly used if RandomCrop is applied
    INPUT_W = 512 # Example, not directly used if RandomCrop is applied
    SCALE_FACTOR = 3 # Resize scale factor for initial loading in dataset
    CROP_H = 512 # Height after RandomCrop
    CROP_W = 512 # Width after RandomCrop
    RESIZE_H = 1024 # Height after Resize transform (model input)
    RESIZE_W = 1024 # Width after Resize transform (model input)
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
        # 512x512の領域をrandom_crop
        A.RandomCrop(height=CROP_H, width=CROP_W, p=1.0),
        # Resize to 1024x1024
        A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_NEAREST),
        # 16の倍数になるようにパディング（min_heightとmin_widthは16の倍数に切り上げ）
        # A.PadIfNeeded(
        #     min_height=16 * ((RESIZE_H + 15) // 16),
        #     min_width=16 * ((RESIZE_W + 15) // 16),
        #     border_mode=cv2.BORDER_CONSTANT
        # ),
        # Add other augmentations here if needed (e.g., Flip, Rotate)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(), # Converts image HWC->CHW, mask HWC->CHW, scales image 0-255 -> 0-1 (mask remains 0 or 255 uint8)
    ])
    
    print(f"Images will be resized to {RESIZE_H}x{RESIZE_W} and padded to ensure dimensions are divisible by 16")
    
       
    # Save masks for debugging
    if os.path.exists(f'/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check'):
        try:
            # Remove existing check directory if it exists
            # 1. Remove existing directory
            shutil.rmtree(f'/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check', ignore_errors=True)
            os.makedirs(f'/workspace/projects/solafune-field-area-segmentation/outputs/{EX_NUM}/check', exist_ok=True) 
        except Exception as e:
            print(f"Error removing existing check directory: {e}")

    # Ensure FieldSegmentationDataset is correctly implemented and paths/file are valid
    try:
        # Initialize dataset with paths, mean/std, and transform
        dataset = FieldSegmentationDataset(
            img_dir=IMAGE_DIR,
            ann_json_path=ANNOTATION_FILE, # Corrected parameter name
            scale_factor=SCALE_FACTOR,     # Pass scale_factor
            transform=transform,
            # edge_width and contact_width use defaults if not specified
            contact_width=5,
            edge_width=3,
            mean=DATASET_MEAN, # Pass pre-calculated mean
            std=DATASET_STD   # Pass pre-calculated std
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

        # --- Inference on Training Data ---
        print("\nStarting inference on training data...")
        PREDICTION_DIR = os.path.join(OUTPUT_DIR, 'train_predictions')
        if not os.path.exists(PREDICTION_DIR):
            os.makedirs(PREDICTION_DIR)
            print(f"Prediction directory created: {PREDICTION_DIR}")

        # Use the same dataset but with batch_size=1 and shuffle=False for inference
        # Re-create transform without random augmentations if needed, but using the same for simplicity here
        inference_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            progress_bar_infer = tqdm(inference_dataloader, desc="Inferring on train set")
            for i, batch in enumerate(progress_bar_infer):
                if batch is None: continue
                imgs, _ = batch # We don't need masks for inference
                imgs = imgs.to(DEVICE)

                # Get original image path to derive output filename and original size
                # This assumes dataset.__getitem__ returns data in the same order as self.img_filenames
                # It's safer to modify the dataset to return the filename or index
                # For simplicity, we reconstruct the path based on index (less robust)
                if i < len(dataset.img_filenames):
                    original_img_filename = dataset.img_filenames[i]
                    original_img_path = os.path.join(IMAGE_DIR, original_img_filename)
                    output_filename_base = os.path.splitext(original_img_filename)[0] + "_pred.png"
                    output_path = os.path.join(PREDICTION_DIR, output_filename_base)

                    # We don't need the original shape for this visualization.
                    # We want to see the prediction corresponding to the 512x512 crop.
                    # The model output corresponds to the 1024x1024 input size.
                    # We will resize the output back to the CROP size (512x512).

                    # Perform inference
                    outputs = model(imgs)
                    outputs = torch.sigmoid(outputs)
                    pred_masks = (outputs > 0.5).float() # Thresholding

                    # Process and save mask
                    pred_mask_np = pred_masks.squeeze(0).cpu().numpy() # (C, H, W) - Size after transform
                    pred_mask_np = pred_mask_np.transpose((1, 2, 0)) # (H, W, C)

                    # Resize mask from model output size (RESIZE_H, RESIZE_W) back to CROP size
                    pred_mask_np = cv2.resize(pred_mask_np, (CROP_W, CROP_H), interpolation=cv2.INTER_NEAREST)
                    # Ensure 3 channels after resize (important if resize outputs 2D)
                    if pred_mask_np.ndim == 2:
                         pred_mask_np = np.stack([pred_mask_np]*3, axis=-1)
                    elif pred_mask_np.ndim == 3 and pred_mask_np.shape[2] == 1:
                         # If resize somehow kept 3 dims but only 1 channel
                         pred_mask_np = np.concatenate([pred_mask_np]*3, axis=-1)

                    # Save the combined mask and per-class masks
                    try:
                        # Save combined mask (optional, might be hard to visualize)
                        # cv2.imwrite(output_path, (pred_mask_np * 255).astype(np.uint8))

                        # Save each class mask separately
                        for class_idx in range(pred_mask_np.shape[2]):
                            class_output_path = os.path.join(PREDICTION_DIR, f"{os.path.splitext(output_filename_base)[0]}_class_{class_idx}.png")
                            cv2.imwrite(class_output_path, (pred_mask_np[:, :, class_idx] * 255).astype(np.uint8))
                        progress_bar_infer.set_postfix(saved=output_filename_base)
                    except Exception as e:
                        print(f"Error saving prediction for {original_img_filename}: {e}")
                else:
                    print(f"Warning: Index {i} out of bounds for dataset filenames.")

        print(f"Training data predictions saved to {PREDICTION_DIR}")


    except NameError:
         print("Error: FieldSegmentationDataset or UNet class not found. Ensure 'src' is in PYTHONPATH or run from the project root. Cannot run training.")
    except FileNotFoundError as e:
         print(f"Error: File or directory not found. Please check paths. Details: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during setup or training: {e}")