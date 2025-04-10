import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
    class UNet(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Linear(1,1)
        def forward(self, x): return self.dummy(torch.zeros(x.shape[0], 1))

def inference(model_path, image_path, output_path, device='cuda'):
    """
    Performs inference on a given image using a trained U-Net model.

    Args:
        model_path (str): Path to the trained model (.pth file).
        image_path (str): Path to the input image (.tif file).
        output_path (str): Path to save the segmentation result.
        device (str): Device to run inference on ('cuda' or 'cpu').
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, switching to CPU.")
        device = 'cpu'

    # --- Configuration ---
    INPUT_H = 512 # Original image height (example)
    INPUT_W = 512 # Original image width (example)
    SCALE_FACTOR = 2 # Resize scale factor from requirements
    RESIZE_H = INPUT_H * SCALE_FACTOR
    RESIZE_W = INPUT_W * SCALE_FACTOR
    # ---------------------

    # 1. Load the model
    model = UNet(backbone_name='maxvit_small_tf_512.in1k', pretrained=False, num_classes=3)  # Adjust parameters as needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # 2. Load and preprocess the image
    try:
        import rasterio
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
            original_shape = (src.height, src.width)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Normalize image (per-image normalization)
        img_mean = image.mean(axis=(1, 2), keepdims=True)
        img_std = image.std(axis=(1, 2), keepdims=True)
        image = (image - img_mean) / (img_std + 1e-6)

        # Transpose image for Albumentations (C, H, W) -> (H, W, C)
        image = image.transpose((1, 2, 0))

        # Define transformations including the required resize
    except ImportError:
        print("Error: rasterio library not found. Please install it using 'pip install rasterio'")
        return
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}")
        return
    
    # MaxViTモデルは入力サイズが16の倍数である必要があるため、それに合わせて調整
    # 各画像のサイズは異なるため、PadIfNeededを使用して16の倍数にパディング
    transform = A.Compose([
        # まず指定サイズにリサイズ
        A.Resize(height=RESIZE_H, width=RESIZE_W, interpolation=cv2.INTER_LINEAR),
        # 16の倍数になるようにパディング（min_heightとmin_widthは16の倍数に切り上げ）
        # A.PadIfNeeded(
        #     min_height=16 * ((RESIZE_H + 15) // 16),
        #     min_width=16 * ((RESIZE_W + 15) // 16),
        #     border_mode=cv2.BORDER_CONSTANT
        # ),
        # Add other augmentations here if needed (e.g., Flip, Rotate)
        # A.HorizontalFlip(p=0.5),
        ToTensorV2(), # Converts image HWC->CHW, mask HWC->CHW, scales image 0-255 -> 0-1 (mask remains 0 or 255 uint8)
    ])

    transformed = transform(image=image)
    image = transformed["image"] # ToTensorV2 already converts to (C, H, W) tensor
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # 3. Perform inference
    with torch.no_grad():  # Disable gradient calculation during inference
        output = model(image)

    # 4. Post-process the output
    # Example: Convert logits to probabilities using sigmoid
    output = torch.sigmoid(output)
    # Example: Threshold the probabilities to get a binary mask
    mask = (output > 0.5).float()

    # 5. Save the result
    # Convert the mask to a numpy array
    mask = mask.squeeze().cpu().numpy() # Shape: (C, H, W)

    # Transpose mask to (H, W, C) for resizing
    mask = mask.transpose((1, 2, 0))

    # Resize mask to original image size
    # original_shape was stored when loading the image
    if 'original_shape' in locals():
         mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
         # Ensure mask remains 3 channels if it started as such after resize
         if mask.ndim == 2 and original_shape[1] > 0 and original_shape[0] > 0: # If resize resulted in 2D mask, stack it
             mask = np.stack([mask]*3, axis=-1)
         elif mask.ndim == 3 and mask.shape[2] == 1: # If resize resulted in (H, W, 1), replicate channel
             mask = np.concatenate([mask]*3, axis=-1)

    # Save the result
    try:
        # Ensure mask is uint8 for saving
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))
        # クラスごとのマスクを保存する場合
        for i in range(mask.shape[2]):
            cv2.imwrite(f"{output_path}_class_{i}.png", (mask[:, :, i] * 255).astype(np.uint8))
        print(f"Output mask saved to {output_path}")
    except Exception as e:
        print(f"Error saving the output mask to {output_path}: {e}")

if __name__ == "__main__":
    # Example usage:
    model_path = '../outputs/ex0/model.path'  # Path to your trained model
    image_path = '/workspace/projects/solafune-field-area-segmentation/data/test_images/test_0.tif'  # Path to your test image
    # image_path = '/workspace/projects/solafune-field-area-segmentation/data/train_images/train_0.tif'  # Path to your test image
    output_path = f'/workspace/projects/solafune-field-area-segmentation/outputs/ex0/output_images/{image_path.split('/')[-1].replace('.tif','')}.png'  # Path to save the segmentation result
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference(model_path, image_path, output_path, device)
    print(f"Inference complete. Segmentation mask saved to {output_path}")