import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import cv2
from shapely.wkt import loads as wkt_loads
from shapely.geometry import mapping # To convert polygon to coordinates
from skimage.morphology import erosion, dilation, footprint_rectangle
from skimage.segmentation import watershed
# from skimage.measure import label as skimage_label # Not used in this implementation

# Helper function to convert COCO segmentation format to mask
def segmentation_to_mask(segmentation, shape):
    """Converts polygon segmentation [x1, y1, x2, y2,...] to a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    # Ensure segmentation is not empty and is a list/tuple
    if not segmentation or not isinstance(segmentation, (list, tuple)):
        print(f"Warning: Invalid segmentation data type: {type(segmentation)}")
        return mask
    # Ensure segmentation list has an even number of elements >= 6 for a polygon
    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
         print(f"Warning: Segmentation list has invalid number of points: {len(segmentation)}")
         return mask

    try:
        points = np.array(segmentation).reshape(-1, 2).round().astype(np.int32)
        if points.shape[0] >= 3: # Need at least 3 points to form a polygon
            cv2.fillPoly(mask, [points], 1)
        else:
             print(f"Warning: Not enough points ({points.shape[0]}) to form a polygon from segmentation.")
    except Exception as e:
        print(f"Error processing segmentation points: {e}. Segmentation data: {segmentation}")
        return np.zeros(shape, dtype=np.uint8) # Return empty mask on error
    return mask

class FieldSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 field segmentation.

    Args:
        img_dir (str): Directory containing TIF image files.
        ann_json_path (str): Path to the COCO-style JSON annotation file.
        edge_width (int): Width parameter for edge mask generation (erosion kernel size).
        contact_width (int): Width parameter for contact mask generation (dilation kernel size).
        transform (callable, optional): Optional transform to be applied on a sample.
                                       Expected to work like Albumentations (input dict {'image': HWC, 'mask': HWC}).
        mean (list or np.array, optional): Pre-calculated mean for each channel for normalization.
        std (list or np.array, optional): Pre-calculated standard deviation for each channel for normalization.
    """
    def __init__(self, img_dir, ann_json_path, edge_width=3, contact_width=3, transform=None, mean=None, std=None):
        self.img_dir = img_dir
        self.edge_width = edge_width
        self.contact_width = contact_width
        self.transform = transform

        # Validate and reshape mean/std if provided
        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
            if self.mean.ndim == 1:
                self.mean = self.mean.reshape(-1, 1, 1)
            elif self.mean.ndim != 3 or self.mean.shape[1:] != (1, 1):
                 raise ValueError("Mean must be a 1D array or a 3D array of shape (C, 1, 1)")
        else:
            self.mean = None

        if std is not None:
            self.std = np.array(std, dtype=np.float32)
            if self.std.ndim == 1:
                self.std = self.std.reshape(-1, 1, 1)
            elif self.std.ndim != 3 or self.std.shape[1:] != (1, 1):
                 raise ValueError("Std must be a 1D array or a 3D array of shape (C, 1, 1)")
        else:
            self.std = None

        if self.mean is not None and self.std is None or self.mean is None and self.std is not None:
            print("Warning: Both mean and std must be provided for pre-calculated normalization. Falling back to per-image normalization.")
            self.mean = None
            self.std = None

        # print(f"Loading annotations from {ann_json_path}...")
        try:
            with open(ann_json_path, 'r') as f:
                ann_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found: {ann_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {ann_json_path}")

        # Validate basic structure
        if 'images' not in ann_data or not isinstance(ann_data['images'], list):
            raise ValueError(f"Invalid annotation format: 'images' key missing or not a list in {ann_json_path}")

        self.annotations = {}
        for item in ann_data['images']:
             if isinstance(item, dict) and 'file_name' in item and 'annotations' in item:
                  # Ensure annotations is a list
                  if isinstance(item['annotations'], list):
                      self.annotations[item['file_name']] = item['annotations']
                  else:
                      print(f"Warning: Skipping image {item['file_name']} due to invalid 'annotations' type (expected list).")
             else:
                  print(f"Warning: Skipping invalid image entry in annotations: {item}")

        # List images in img_dir and filter by those present in annotations
        try:
            all_files = os.listdir(img_dir)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.img_filenames = sorted([
            fn for fn in all_files
            if fn.endswith('.tif') and fn in self.annotations
        ])
        print(f"Found {len(self.img_filenames)} images in {img_dir} with corresponding annotations.")

        if not self.img_filenames:
             print(f"Warning: No matching .tif files found in {img_dir} that are listed in {ann_json_path}")
             # Depending on use case, might want to raise error or allow empty dataset
             # raise ValueError(f"No matching image files found in {img_dir} for annotations in {ann_json_path}")


    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if idx >= len(self.img_filenames):
             raise IndexError("Dataset index out of range")

        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        # Load 12-band image
        try:
            with rasterio.open(img_path) as src:
                # Check channel count if needed, but proceed anyway
                # num_channels_actual = src.count
                # if self.mean is not None and num_channels_actual != self.mean.shape[0]:
                #     print(f"Warning: Image {img_filename} has {num_channels_actual} channels, but mean/std provided for {self.mean.shape[0]} channels.")
                img = src.read().astype(np.float32) # (C, H, W)
                img_shape = (src.height, src.width)
        except rasterio.RasterioIOError as e:
            print(f"Error loading image {img_path}: {e}")
            # Handle error: return None, skip, or raise. Raising is safer for training loop stability.
            raise IOError(f"Could not read image file {img_path}") from e
        except Exception as e: # Catch other potential errors during file access/read
            print(f"Unexpected error loading image {img_path}: {e}")
            raise IOError(f"Could not read image file {img_path}") from e

        # Ensure image has 3 dimensions (C, H, W)
        if img.ndim != 3 or img.shape[1] != img_shape[0] or img.shape[2] != img_shape[1]:
            raise ValueError(f"Image loaded from {img_path} has unexpected shape: {img.shape}, expected (C, {img_shape[0]}, {img_shape[1]})")

        num_channels = img.shape[0]

        # Normalize each band
        if self.mean is not None and self.std is not None:
            if self.mean.shape[0] != num_channels or self.std.shape[0] != num_channels:
                 # This case should ideally be caught during init or file listing if possible,
                 # but double-check here.
                 raise ValueError(f"Pre-calculated mean/std channel count ({self.mean.shape[0]}) doesn't match image channels ({num_channels}) for {img_filename}")
            # Mean/std should already be shaped (C, 1, 1) from __init__
            img = (img - self.mean) / (self.std + 1e-6)
        else:
            # Per-image normalization (fallback)
            img_mean = img.mean(axis=(1, 2), keepdims=True)
            img_std = img.std(axis=(1, 2), keepdims=True)
            # Add epsilon to std to avoid division by zero
            img = (img - img_mean) / (img_std + 1e-6)

        # --- Mask Generation ---
        img_annotations = self.annotations.get(img_filename, []) # Use .get for safety
        labels = np.zeros(img_shape, dtype=np.uint16) # Use uint16 for potentially many instances
        instance_id = 0
        valid_polygons_found = False
        for ann in img_annotations:
            # Check if annotation is a dict and has required keys
            if not isinstance(ann, dict) or 'class' not in ann or 'segmentation' not in ann:
                print(f"Warning: Skipping invalid annotation structure in {img_filename}: {ann}")
                continue

            if ann['class'] == 'field' and ann['segmentation']:
                poly_mask = np.zeros(img_shape, dtype=np.uint8)
                try:
                    if isinstance(ann['segmentation'], str): # Assume WKT string
                        polygon = wkt_loads(ann['segmentation'])
                        # Handle MultiPolygon as well, iterate through geometries
                        if polygon.geom_type == 'MultiPolygon':
                            all_coords = []
                            for poly in polygon.geoms:
                                coords = np.array(mapping(poly)['coordinates'][0]).round().astype(np.int32) # Get exterior coords
                                if coords.shape[0] >= 3:
                                    all_coords.append(coords)
                            if all_coords:
                                cv2.fillPoly(poly_mask, all_coords, 1)
                        elif polygon.geom_type == 'Polygon':
                             coords = np.array(mapping(polygon)['coordinates'][0]).round().astype(np.int32) # Get exterior coords
                             if coords.shape[0] >= 3:
                                 cv2.fillPoly(poly_mask, [coords], 1)
                        else:
                            print(f"Warning: Unsupported geometry type '{polygon.geom_type}' in WKT for {img_filename}")

                    elif isinstance(ann['segmentation'], (list, tuple)): # Handle COCO list format as fallback
                         # Use existing helper, but ensure it returns 0/1
                         temp_mask = segmentation_to_mask(ann['segmentation'], img_shape)
                         poly_mask[temp_mask > 0] = 1 # Ensure 0/1 output
                    else:
                         print(f"Warning: Unsupported segmentation format for annotation in {img_filename}: {type(ann['segmentation'])}")

                    if np.any(poly_mask): # Only assign label if mask is not empty
                        instance_id += 1 # Increment ID only for valid, non-empty polygons
                        # Assign unique ID only where the current polygon mask is 1 AND no previous label exists
                        # This prevents overwriting labels if polygons overlap slightly after rasterization
                        labels[(poly_mask > 0) & (labels == 0)] = instance_id
                        valid_polygons_found = True
                except Exception as e:
                    print(f"Error processing WKT/segmentation for annotation in {img_filename}: {e}. Data: {ann['segmentation']}")
                # else: poly_mask is empty or error occurred, don't assign label or increment id

        # If no valid polygons were found for this image, masks will be all zeros.
        # if not valid_polygons_found:
        #     print(f"Warning: No valid 'field' polygons found for {img_filename}. Proceeding with empty masks.")

        # 1. Field mask (footprint)
        field_mask = (labels > 0).astype(np.uint8)

        # 2. Edge mask
        edge_mask = np.zeros(img_shape, dtype=np.uint8)
        if self.edge_width > 0 and instance_id > 0: # Check instance_id > 0 (actual number of instances)
            selem_edge = footprint_rectangle((self.edge_width, self.edge_width))
            for i in range(1, instance_id + 1): # Iterate up to the actual number of instances processed
                instance_mask = (labels == i)
                if np.any(instance_mask): # Check if the instance exists and is not empty
                    try:
                        eroded_mask = erosion(instance_mask, selem_edge)
                        edge = instance_mask ^ eroded_mask
                        edge_mask[edge] = 1
                    except Exception as e:
                        print(f"Warning: Error during edge erosion for instance {i} in {img_filename}: {e}")
                        # Continue processing other instances/masks

        # 3. Contact mask
        contact_mask = np.zeros(img_shape, dtype=np.uint8)
        # Need at least 2 actual instances processed for contact
        if self.contact_width > 0 and instance_id > 1:
            selem_contact = footprint_rectangle((self.contact_width, self.contact_width))
            try:
                # Dilate the field mask to find potential contact zones
                dilated_field = dilation(field_mask, selem_contact)

                # Use watershed to separate close objects based on instance labels
                # Ensure labels don't have 0 where dilated_field is True, watershed needs markers > 0
                markers = labels.copy()
                markers[~dilated_field] = 0 # Clear markers outside dilated area

                # Watershed segmentation. watershed_line=True marks boundaries between labels.
                # The mask=dilated_field ensures watershed only runs within the dilated area.
                # Handle potential errors during watershed
                ws_labels = watershed(dilated_field, markers, mask=dilated_field, watershed_line=True)

                # Watershed lines are marked as 0 in ws_labels.
                # Contact points are where the watershed line exists AND it separates different original labels.
                watershed_lines = (ws_labels == 0) & dilated_field

                # Refine: Check neighbors around watershed lines in the original labels
                contact_mask_candidates = watershed_lines
                coords = np.argwhere(contact_mask_candidates)
                for y, x in coords:
                     # Check 3x3 neighborhood in original labels
                     y_min, y_max = max(0, y - 1), min(img_shape[0], y + 2)
                     x_min, x_max = max(0, x - 1), min(img_shape[1], x + 2)
                     neighborhood = labels[y_min:y_max, x_min:x_max]
                     # Get unique non-zero labels in the neighborhood
                     unique_labels_in_neighborhood = np.unique(neighborhood[neighborhood > 0])
                     if len(unique_labels_in_neighborhood) > 1:
                         contact_mask[y, x] = 1
            except Exception as e:
                print(f"Warning: Error during contact mask generation for {img_filename}: {e}")
                # contact_mask remains zeros if an error occurs

        # Stack masks: (C, H, W) format for PyTorch
        # Ensure all masks have the same shape before stacking
        if not (field_mask.shape == edge_mask.shape == contact_mask.shape == img_shape):
            # This should not happen if logic is correct, but check for safety
            raise ValueError(f"Mask shape mismatch for {img_filename}: "
                             f"Field={field_mask.shape}, Edge={edge_mask.shape}, Contact={contact_mask.shape}, Expected={img_shape}")

        # Scale masks to 0/255 as required
        mask = np.stack([field_mask, edge_mask, contact_mask], axis=0).astype(np.uint8) * 255 # Shape: (3, H, W)

        # Apply transformations (e.g., Albumentations)
        if self.transform:
            try:
                # print(f"Applying transformations for {img_filename}...")
                # Albumentations expects image: (H, W, C), mask: (H, W, C) or (H, W, N)
                img_for_transform = img.transpose((1, 2, 0)) # (H, W, C) - NumPyスタイルの転置
                mask_for_transform = mask.transpose((1, 2, 0)) # (H, W, 3) - NumPyスタイルの転置
                # print(f"Input shapes for transform: image={img_for_transform.shape}, mask={mask_for_transform.shape}")
                augmented = self.transform(image=img_for_transform, mask=mask_for_transform)

                img = augmented['image'] # Already (C, H, W) tensor from ToTensorV2
                mask = augmented['mask'] # Already (C, H, W) tensor from ToTensorV2

                # # 変換後のサイズをログに出力して確認
                # print(f"After transform: image shape={img.shape}, mask shape={mask.shape}")
                # 16の倍数かどうかを確認
                if img.shape[1] % 16 != 0 or img.shape[2] % 16 != 0:
                    print(f"Warning: Transformed image dimensions ({img.shape[1]}x{img.shape[2]}) are not divisible by 16")
                
                # ToTensorV2の後、imgとmaskはすでにPyTorchテンソル
                # テンソルの型変換にはto()メソッドを使用
                if isinstance(mask, torch.Tensor):
                    # マスクの形状を確認し、必要に応じてCHW形式に変換
                    # print(f"Before conversion - mask shape: {mask.shape}, type: {mask.dtype}")
                    if mask.ndim == 3 and mask.shape[0] != 3 and mask.shape[2] == 3:  # HWC形式の場合
                        print(f"Converting mask from HWC to CHW format: {mask.shape}")
                        mask = mask.permute(2, 0, 1)  # HWC -> CHW
                        print(f"After permute - mask shape: {mask.shape}")
                    mask = mask.to(torch.uint8)  # テンソルの場合はto()を使用
                    print(f"Final mask shape: {mask.shape}, type: {mask.dtype}")
                else:
                    print(f"Mask is NumPy array with shape: {mask.shape}, type: {mask.dtype}")
                    mask = mask.astype(np.uint8)  # NumPy配列の場合はastype()を使用
                
                if isinstance(img, torch.Tensor):
                    # 画像の形状を確認し、必要に応じてCHW形式に変換
                    print(f"Before conversion - image shape: {img.shape}, type: {img.dtype}")
                    if img.ndim == 3 and img.shape[0] != 12 and img.shape[2] == 12:  # HWC形式の場合
                        print(f"Converting image from HWC to CHW format: {img.shape}")
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                        print(f"After permute - image shape: {img.shape}")
                    img = img.to(torch.float32)  # テンソルの場合はto()を使用
                    print(f"Final image shape: {img.shape}, type: {img.dtype}")
                else:
                    print(f"Image is NumPy array with shape: {img.shape}, type: {img.dtype}")
                    img = img.astype(np.float32)  # NumPy配列の場合はastype()を使用

            except Exception as e:
                print(f"Error during transform application for {img_filename}: {e}")
                # Decide how to handle transform errors: skip transform, raise error?
                # For now, let's proceed with untransformed data if transform fails, but log warning.
                print(f"Warning: Proceeding with untransformed data for {img_filename} due to transform error.")
                # Ensure original mask is used if transform failed mid-way
                mask = np.stack([field_mask, edge_mask, contact_mask], axis=0).astype(np.uint8)
                # img is already normalized before transform attempt

        # Convert to tensors if they aren't already
        try:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img.copy()).float()
            else:
                img_tensor = img.clone()  # すでにテンソルの場合はclone()を使用
                
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask.copy()).to(torch.uint8)
            else:
                mask_tensor = mask.clone()  # すでにテンソルの場合はclone()を使用
        except Exception as e:
            print(f"Error converting numpy arrays to tensors for {img_filename}: {e}")
            raise TypeError(f"Could not convert data to tensors for {img_filename}") from e

        # Final check for tensor shapes
        if img_tensor.shape[1:] != mask_tensor.shape[1:]:
             raise ValueError(f"Final tensor shape mismatch for {img_filename}: "
                              f"Image shape {img_tensor.shape}, Mask shape {mask_tensor.shape}")

        # # 最終的なテンソルの形状をログに出力
        # print(f"Final tensors for {img_filename}:")
        # print(f"  Image tensor: shape={img_tensor.shape}, type={img_tensor.dtype}")
        # print(f"  Mask tensor: shape={mask_tensor.shape}, type={mask_tensor.dtype}")

        return img_tensor, mask_tensor