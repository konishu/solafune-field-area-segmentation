import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import cv2 # Import OpenCV
import os

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
    class UNet(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = torch.nn.Linear(1,1)
        def forward(self, x): return self.dummy(torch.zeros(x.shape[0], 1))


if __name__ == "__main__":
    # --- Configuration (Should match the training configuration used to save the model) ---
    ROOT_DIR = '/workspace/projects/solafune-field-area-segmentation'
    EX_NUM = 'ex0' # Example experiment number
    IMAGE_DIR = os.path.join(ROOT_DIR, 'data/inference_images') # Path to training images used for inference
    ANNOTATION_FILE = os.path.join(ROOT_DIR, 'data/train_annotation.json') # Needed for dataset initialization
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', EX_NUM, 'check') # Directory where the model is saved
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.path') # Path to the saved model state dict
    PREDICTION_DIR = os.path.join(OUTPUT_DIR, 'train_predictions_inference_script') # Output directory for predictions from this script
    BACKBONE = 'maxvit_small_tf_512.in1k' # Must match the trained model's backbone
    NUM_OUTPUT_CHANNELS = 3 # Must match the trained model's output channels
    PRETRAINED = False # Pretrained weights are loaded from MODEL_PATH, not downloaded again
    NUM_WORKERS = 4 # Adjust based on CPU cores
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SCALE_FACTOR = 3 # Must match dataset settings during training
    CROP_H = 512 # Must match dataset settings during training
    CROP_W = 512 # Must match dataset settings during training
    RESIZE_H = 1024 # Must match model input size during training
    RESIZE_W = 1024 # Must match model input size during training
    DATASET_MEAN = None # Use the same mean/std as during training
    DATASET_STD = None  # Use the same mean/std as during training
    # ---------------------

    print("Setting up dataset for inference...")
    # Define transformations (should be consistent with training, but without random augmentations if desired)
    # Using the same transform as training for simplicity here, but RandomCrop might not be ideal for inference
    # A deterministic crop or resize might be better depending on the goal.
    transform = A.Compose([
        A.RandomCrop(height=CROP_H, width=CROP_W, p=1.0), # Note: RandomCrop might not be ideal for inference
        A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_NEAREST),
        # A.PadIfNeeded(...) # Add padding if used during training
        ToTensorV2(),
    ])

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
            contact_width=5, # Match training settings
            edge_width=3,    # Match training settings
            mean=DATASET_MEAN,
            std=DATASET_STD
        )

        if len(dataset) == 0:
             print(f"Error: Dataset is empty. Check image path '{IMAGE_DIR}' and annotation file '{ANNOTATION_FILE}'.")
             exit()
        print(f"Dataset initialized with {len(dataset)} samples.")

        # Use DataLoader with batch_size=1 and shuffle=False for inference
        inference_dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)
        print("Dataloader ready for inference.")

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

        # --- Inference ---
        print("\nStarting inference...")
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            progress_bar_infer = tqdm(inference_dataloader, desc="Inferring")
            for batch_idx, batch in enumerate(progress_bar_infer):
                if batch is None:
                    continue
                imgs, _ = batch
                imgs = imgs.to(DEVICE)

                # Perform inference
                outputs = model(imgs)
                outputs = torch.sigmoid(outputs)
                print("try to get outputs")
                # Apply different thresholds for each class
                thresholds = [0.5, 0.1, 0.1]  # Example thresholds for classes 0, 1, and 2
                pred_masks = torch.zeros_like(outputs)
                for class_idx, threshold in enumerate(thresholds):
                    pred_masks[:, class_idx, :, :] = (outputs[:, class_idx, :, :] > threshold).float()

                # Process each image in the batch individually
                batch_size = imgs.shape[0]
                for b in range(batch_size):
                    dataset_idx = batch_idx * inference_dataloader.batch_size + b
                    if dataset_idx < len(dataset.img_filenames):
                        original_img_filename = dataset.img_filenames[dataset_idx]
                        output_filename_base = os.path.splitext(original_img_filename)[0] + "_pred.png"

                        # Get the predicted mask for this image
                        pred_mask_np = pred_masks[b].cpu().numpy()  # (C, H, W)
                        pred_mask_np = np.transpose(pred_mask_np, (1, 2, 0))  # (H, W, C)

                        # Resize mask from model output size (RESIZE_H, RESIZE_W) back to CROP size
                        pred_mask_np = cv2.resize(pred_mask_np, (CROP_W, CROP_H), interpolation=cv2.INTER_NEAREST)

                        # Ensure 3 channels after resize
                        if pred_mask_np.ndim == 2:
                            pred_mask_np = np.stack([pred_mask_np] * 3, axis=-1)
                        elif pred_mask_np.ndim == 3 and pred_mask_np.shape[2] == 1:
                            pred_mask_np = np.concatenate([pred_mask_np] * 3, axis=-1)

                        # Save each class mask separately
                        try:
                            for class_idx in range(pred_mask_np.shape[2]):
                                class_output_path = os.path.join(PREDICTION_DIR, f"{os.path.splitext(output_filename_base)[0]}_class_{class_idx}.png")
                                # Ensure the mask is uint8 for saving
                                mask_to_save = (pred_mask_np[:, :, class_idx] * 255).astype(np.uint8)
                                cv2.imwrite(class_output_path, mask_to_save)
                            # 3-channel maskを1つにまとめて保存。(class_0:Blue, class_1:Green, class_2:Red)
                            combined_mask = np.zeros_like(pred_mask_np)
                            combined_mask[:, :, 0] = (pred_mask_np[:, :, 0] * 255).astype(np.uint8)  # Blue channel
                            combined_mask[:, :, 1] = (pred_mask_np[:, :, 1] * 255).astype(np.uint8)  # Green channel
                            combined_mask[:, :, 2] = (pred_mask_np[:, :, 2] * 255).astype(np.uint8)  # Red channel
                            # Save the combined mask
                            combined_output_path = os.path.join(PREDICTION_DIR, f"{os.path.splitext(output_filename_base)[0]}_combined.png")
                            cv2.imwrite(combined_output_path, combined_mask)
                            print(f"Output mask saved to {class_output_path} and {combined_output_path}")

                            progress_bar_infer.set_postfix(saved=output_filename_base)
                        except Exception as e:
                            print(f"Error saving prediction for {original_img_filename}: {e}")
                    else:
                        print(f"Warning: Index {dataset_idx} out of bounds for dataset filenames.")

        print(f"\nInference complete. Predictions saved to {PREDICTION_DIR}")

    except NameError:
         print("Error: FieldSegmentationDataset or UNet class not found. Ensure 'src' is in PYTHONPATH or run from the project root.")
    except FileNotFoundError as e:
         print(f"Error: File or directory not found. Please check paths. Details: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during setup or inference: {e}")