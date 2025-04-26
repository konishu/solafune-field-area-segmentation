import json
import os

import cv2
import numpy as np
import rasterio
import torch
from shapely.geometry import MultiPolygon, Polygon, mapping  # To convert polygon to coordinates
from shapely.wkt import loads as wkt_loads
from skimage.morphology import (
    dilation,
    erosion,
    footprint_rectangle,
)  # Using footprint_rectangle instead of deprecated square

# footprint_rectangle is also fine, choose one consistently
# from skimage.morphology import footprint_rectangle
from skimage.segmentation import watershed
from torch.utils.data import Dataset

# from skimage.measure import label as skimage_label # Not used in this implementation

def fill_nan_pixels(img):
    """Fill NaN pixels with the mean of their valid neighbors."""
    for c in range(img.shape[0]):  # 各チャネルについて
        nan_mask = np.isnan(img[c])
        if np.any(nan_mask):  # NaNピクセルが存在する場合
            coords = np.argwhere(nan_mask)
            for y, x in coords:
                # 有効な近傍ピクセルの値を取得
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < img.shape[1] and 0 <= nx < img.shape[2] and not np.isnan(img[c, ny, nx]):
                            neighbors.append(img[c, ny, nx])
                # 近傍ピクセルの平均値で補完 (近傍がない場合は0で補完)
                if neighbors:
                    img[c, y, x] = np.mean(neighbors)
                else:
                    img[c, y, x] = 0
    return img

# Helper function to convert COCO segmentation format to mask
def segmentation_to_mask(segmentation, shape, scale_factor=1.0):
    """Converts polygon segmentation [x1, y1, x2, y2,...] to a binary mask, applying scaling."""
    mask = np.zeros(shape, dtype=np.uint8)
    # Ensure segmentation is not empty and is a list/tuple
    if not segmentation or not isinstance(segmentation, (list, tuple)):
        # print(f"Warning: Invalid segmentation data type: {type(segmentation)}") # Reduced verbosity
        return mask
    # Ensure segmentation list has an even number of elements >= 6 for a polygon
    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
        # print(f"Warning: Segmentation list has invalid number of points: {len(segmentation)}") # Reduced verbosity
        return mask

    try:
        # Apply scale factor before rounding
        points = (np.array(segmentation).reshape(-1, 2) * scale_factor).round().astype(np.int32)
        if points.shape[0] >= 3:  # Need at least 3 points to form a polygon
            cv2.fillPoly(mask, [points], 1)
        # else: # Reduced verbosity
        # print(f"Warning: Not enough points ({points.shape[0]}) to form a polygon from segmentation.")
    except Exception as e:
        print(f"Error processing segmentation points: {e}. Segmentation data: {segmentation}")
        return np.zeros(shape, dtype=np.uint8)  # Return empty mask on error
    return mask


class FieldSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 field segmentation.
    Includes modified contact mask generation with boundary dilation.

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

    def __init__(
        self,
        img_dir,
        cache_dir,
        ann_json_path,
        scale_factor=1.0,
        edge_width=3,
        contact_width=3,
        transform=None,
        img_idxes = None,
        mean=None,
        std=None,
    ):
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.img_idxes = img_idxes
        self.scale_factor = scale_factor
        self.edge_width = edge_width
        self.contact_width = contact_width
        self.transform = transform

        # --- (Initialization code remains the same) ---
        if mean is not None:
            self.mean = np.array(mean, dtype=np.float32)
            if self.mean.ndim == 1:
                self.mean = self.mean.reshape(-1, 1, 1)
            elif self.mean.ndim != 3 or self.mean.shape[1:] != (1, 1):
                raise ValueError("Mean shape error")
        else:
            self.mean = None
        if std is not None:
            self.std = np.array(std, dtype=np.float32)
            if self.std.ndim == 1:
                self.std = self.std.reshape(-1, 1, 1)
            elif self.std.ndim != 3 or self.std.shape[1:] != (1, 1):
                raise ValueError("Std shape error")
        else:
            self.std = None
        if (self.mean is None) != (self.std is None):
            print("Warning: Provide both mean and std, or neither. Falling back.")
            self.mean, self.std = None, None

        try:
            with open(ann_json_path) as f:
                ann_data = json.load(f)
        except Exception as e:
            raise OSError(f"Error reading annotation file {ann_json_path}: {e}") from e

        if "images" not in ann_data or not isinstance(ann_data["images"], list):
            raise ValueError(f"Invalid annotation format in {ann_json_path}")

        self.annotations = {}
        for item in ann_data.get("images", []):  # Use .get for safety
            if isinstance(item, dict) and "file_name" in item and "annotations" in item:
                if isinstance(item["annotations"], list):
                    self.annotations[item["file_name"]] = item["annotations"]
                # else: # Reduced verbosity
                # print(f"Warning: Skipping {item['file_name']}, invalid annotations type.")
            # else: # Reduced verbosity
            # print(f"Warning: Skipping invalid image entry: {item}")

        try:
            if img_idxes is None:
                all_files = os.listdir(img_dir)
            else:
                all_files = [f'train_{idx}.tif' for idx in self.img_idxes]
        except FileNotFoundError:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.img_filenames = sorted([fn for fn in all_files if fn.endswith(".tif") and fn in self.annotations])
        print(f"Found {len(self.img_filenames)} images in {img_dir} with annotations.")
        if not self.img_filenames:
            print(f"Warning: No matching .tif files found in {img_dir} listed in {ann_json_path}")
        
        if os.path.exists(self.cache_dir):
            print(f"Cache directory exists: {self.cache_dir}")
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # --- (End of Initialization code) ---

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if idx >= len(self.img_filenames):
            raise IndexError("Dataset index out of range")

        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        cache_path = os.path.join(self.cache_dir, img_filename.replace(".tif", ".npz"))
        try:
            # print(f"Loading from cache: {cache_path}")
            loaded = np.load(cache_path)
            img = loaded["img"]
            mask = loaded["mask"]
            num_channels = img.shape[0]
            original_height, original_width = img.shape[1:]
            img_shape = original_height, original_width

            field_mask = mask[0]

        except:
            print(f"Loading image {img_path} (and creating cache)...")

            # Load 12-band image
            try:
                with rasterio.open(img_path) as src:
                    img = src.read().astype(np.float32)  # (C, H, W)
                    img_shape = (src.height, src.width)
                    
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                raise OSError(f"Could not read image file {img_path}") from e

            if img.ndim != 3 or img.shape[1:] != img_shape:
                raise ValueError(
                    f"Image {img_path} has unexpected shape: {img.shape}, expected (C, {img_shape[0]}, {img_shape[1]})"
                )
            
            # train_38,42にnanがあるので補完
            nan_mask = True
            for c in range(img.shape[0]):
                nan_mask = np.isnan(img[c]) * nan_mask
            if np.any(nan_mask):
                print(f"Warning: NaN pixels found in {img_path}. Filling with mean of neighbors.")  
                img = fill_nan_pixels(img) 
            
            num_channels = img.shape[0]
            original_height, original_width = img_shape

            # --- Stage 1 Resize based on scale_factor ---
            if self.scale_factor != 1.0:
                target_h = int(original_height * self.scale_factor)
                target_w = int(original_width * self.scale_factor)
                img_hwc = img.transpose((1, 2, 0))
                img_resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                if img_resized_hwc.ndim == 2:
                    img_resized_hwc = np.expand_dims(img_resized_hwc, axis=-1)
                img = img_resized_hwc.transpose((2, 0, 1))
                img_shape = (target_h, target_w)  # Update shape for masks

            # target_h = int(original_height * self.scale_factor)
            # target_w = int(original_width * self.scale_factor)
            # img_shape = (target_h, target_w)
            # print(f"Using original image shape: C={img.shape[0]}, H={img_shape[0]}, W={img_shape[1]}")

            # --- Normalize image ---
            if self.mean is not None and self.std is not None:
                if self.mean.shape[0] != num_channels or self.std.shape[0] != num_channels:
                    raise ValueError(f"Mean/std channel mismatch for {img_filename}")
                img = (img - self.mean) / (self.std + 1e-6)
            else:
                img_mean = img.mean(axis=(1, 2), keepdims=True)
                img_std = img.std(axis=(1, 2), keepdims=True)
                img = (img - img_mean) / (img_std + 1e-6)

            # --- Mask Generation ---
            img_annotations = self.annotations.get(img_filename, [])
            labels = np.zeros(img_shape, dtype=np.uint16)
            instance_id = 0
            # --- (Instance Label Map Generation - same as before) ---
            for ann in img_annotations:
                if not isinstance(ann, dict) or "class" not in ann or "segmentation" not in ann:
                    continue
                if ann["class"] == "field" and ann["segmentation"]:
                    poly_mask = np.zeros(img_shape, dtype=np.uint8)
                    try:
                        # --- WKT/COCO Polygon Processing (same as before) ---
                        if isinstance(ann["segmentation"], str):  # WKT
                            polygon = wkt_loads(ann["segmentation"])
                            coords_list = []
                            if polygon.geom_type == "MultiPolygon":
                                for poly in polygon.geoms:
                                    coords = np.array(mapping(poly)["coordinates"][0]).round().astype(np.int32)
                                    if coords.shape[0] >= 3:
                                        coords_list.append(coords)
                            elif polygon.geom_type == "Polygon":
                                coords = np.array(mapping(polygon)["coordinates"][0]).round().astype(np.int32)
                                if coords.shape[0] >= 3:
                                    coords_list.append(coords)
                            if coords_list:
                                cv2.fillPoly(poly_mask, coords_list, 1)
                        elif isinstance(ann["segmentation"], (list, tuple)):  # COCO list
                            # check if segmentation is suit as Polygon
                            try:
                                # フラットリストを (x, y) のペアに変換
                                seg = ann["segmentation"]
                                if all(isinstance(x, (int, float)) for x in seg):
                                    if len(seg) % 2 != 0:
                                        raise ValueError("Segmentation list must contain even number of coordinates.")
                                    coords = list(zip(seg[0::2], seg[1::2]))  # [(x1, y1), (x2, y2), ...]
                                    polygon = Polygon(coords)
                                else:
                                    # すでに [[x1, y1, x2, y2, ...]] 形式の場合（複数ポリゴン）
                                    polygon = MultiPolygon([Polygon(list(zip(poly[0::2], poly[1::2]))) for poly in seg])
                            except Exception as e:
                                print(f"Warning: Invalid segmentation data for {img_filename}: {e}")
                                print(ann["segmentation"])
                                continue
                            # exit()
                            # Pass the scale_factor from the dataset instance
                            temp_mask = segmentation_to_mask(ann["segmentation"], img_shape, self.scale_factor)
                            poly_mask[temp_mask > 0] = 1
                        # --- End WKT/COCO ---
                        if np.any(poly_mask):
                            instance_id += 1
                            labels[(poly_mask > 0) & (labels == 0)] = instance_id
                    except Exception as e:
                        print(
                            f"Error processing annotation for {img_filename}: {e}. Data: {ann.get('segmentation', 'N/A')}"
                        )  # Use .get for safety
            # --- (End Instance Label Map Generation) ---
            # 1. Field mask (footprint)
            field_mask = (labels > 0).astype(np.uint8)

            # 2. Edge mask
            edge_mask = np.zeros(img_shape, dtype=np.uint8)
            if self.edge_width > 0 and instance_id > 0:
                # Use square kernel for consistency with process_image logic if desired
                selem_edge = footprint_rectangle((self.edge_width, self.edge_width))
                # Or use footprint_rectangle as before:
                # from skimage.morphology import footprint_rectangle
                # selem_edge = footprint_rectangle((self.edge_width, self.edge_width))
                for i in range(1, instance_id + 1):
                    instance_mask = labels == i
                    if np.any(instance_mask):
                        try:
                            eroded_mask = erosion(instance_mask, selem_edge)
                            edge = instance_mask ^ eroded_mask  # XOR
                            edge_mask[edge] = 1  # Accumulate edges
                        except Exception as e:
                            print(f"Warning: Error during edge erosion for instance {i} in {img_filename}: {e}")

            # 3. Contact mask (Modified Logic)
            contact_mask = np.zeros(img_shape, dtype=np.uint8)
            if self.contact_width > 0 and instance_id > 1:  # Need >= 2 instances
                # Use square kernel for consistency if edge_mask used square
                selem_contact = footprint_rectangle((self.contact_width, self.contact_width))
                # Or use footprint_rectangle as before:
                # from skimage.morphology import footprint_rectangle
                # selem_contact = footprint_rectangle((self.contact_width, self.contact_width))
                try:
                    # Dilate the field mask
                    dilated_field = dilation(field_mask, selem_contact)

                    # Prepare markers for watershed
                    markers = labels.copy()
                    markers[~dilated_field] = 0  # Clear markers outside dilated area

                    # Run watershed
                    ws_labels = watershed(dilated_field, markers, mask=dilated_field, watershed_line=True)

                    # Identify watershed lines (potential boundaries)
                    watershed_lines = (ws_labels == 0) & dilated_field  # Boolean mask

                    # >>> MODIFICATION START <<<
                    # Combine watershed lines with edge mask
                    # Ensure edge_mask is boolean for bitwise OR
                    combined_boundaries = watershed_lines | (edge_mask > 0)

                    # Dilate the combined boundaries
                    dilated_boundaries = dilation(combined_boundaries, selem_contact)

                    # Use the dilated boundaries as candidates for contact points
                    contact_mask_candidates = dilated_boundaries
                    # >>> MODIFICATION END <<<

                    # Verify candidates by checking neighbors in the original labels map
                    coords = np.argwhere(contact_mask_candidates)  # Use the new candidates
                    for y, x in coords:
                        # Check 3x3 neighborhood in original labels
                        y_min, y_max = max(0, y - 1), min(img_shape[0], y + 2)
                        x_min, x_max = max(0, x - 1), min(img_shape[1], x + 2)
                        neighborhood = labels[y_min:y_max, x_min:x_max]
                        # Get unique non-zero labels in the neighborhood
                        unique_labels_in_neighborhood = np.unique(neighborhood[neighborhood > 0])
                        # If more than one unique label exists, it's a contact point
                        if len(unique_labels_in_neighborhood) > 1:
                            contact_mask[y, x] = 1
                except Exception as e:
                    print(f"Warning: Error during contact mask generation for {img_filename}: {e}")
                    # contact_mask remains zeros if an error occurs

            # --- Stack and finalize masks ---
            if not (field_mask.shape == edge_mask.shape == contact_mask.shape == img_shape):
                raise ValueError(f"Mask shape mismatch for {img_filename}")

            # # Save masks for debugging
            # cv2.imwrite(f'/workspace/projects/solafune-field-area-segmentation/outputs/ex0/check/field_{img_path.split("/")[-1].replace(".tif","")}.png', field_mask * 255)  # Save field mask as PNG (0-255)
            # cv2.imwrite(f'/workspace/projects/solafune-field-area-segmentation/outputs/ex0/check/edge_{img_path.split('/')[-1].replace(".tif","")}.png', edge_mask * 255)  # Save edge mask as PNG (0-255)
            # cv2.imwrite(f'/workspace/projects/solafune-field-area-segmentation/outputs/ex0/check/contact_{img_path.split("/")[-1].replace(".tif","")}.png', contact_mask * 255)  # Save contact mask as PNG (0-255)
            # Stack masks: (3, H, W)
            mask = np.stack([field_mask, edge_mask, contact_mask], axis=0).astype(np.uint8)
            # 3クラスのマスクを作成 (0: 背景, 1: field, 2: contact, 3: edge)し、red, blue, greenにして保存
            mask_ = np.stack([field_mask, contact_mask, edge_mask], axis=0).astype(np.uint8)  # (3, H, W)
            # Save the combined mask as PNG (0-255)
            cv2.imwrite(
                f"/workspace/projects/solafune-field-area-segmentation/outputs/ex0/check/combined_{img_path.split('/')[-1].replace('.tif', '')}.png",
                mask_.transpose((1, 2, 0)) * 255,
            )  # Save combined mask as PNG (0-255)

            # --- Save to cache ---
            np.savez_compressed(cache_path, img=img, mask=mask)

        # --- Resize masks if image was resized (using updated img_shape) ---
        if self.scale_factor != 1.0:
            target_h, target_w = img_shape  # Already updated
            resized_mask_channels = []
            for i in range(mask.shape[0]):
                # Use INTER_NEAREST for masks to preserve 0/1 values
                resized_ch = cv2.resize(mask[i], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                resized_mask_channels.append(resized_ch)
            mask = np.stack(resized_mask_channels, axis=0)
        # print("Final mask shape:", mask.shape, f"{ img_shape= }")

        # Scale mask values to 0-255 AFTER potential resizing
        mask = mask * 255
        # cv2.imwrite(f'/workspace/projects/solafune-field-area-segmentation/outputs/ex0/check/train_{img_path.split("/")[-1].replace(".tif","")}.png', mask.transpose((1, 2, 0)) )  # Save combined mask as PNG (0-255)

        # --- Apply transformations ---
        if self.transform:
            try:
                img_for_transform = img.transpose((1, 2, 0))  # HWC for Albumentations
                mask_for_transform = mask.transpose((1, 2, 0))  # HWC for Albumentations

                augmented = self.transform(image=img_for_transform, mask=mask_for_transform)

                img = augmented["image"]  # Expecting Tensor CHW from ToTensorV2
                mask = augmented["mask"]  # Expecting Tensor CHW from ToTensorV2

                # Ensure mask is uint8 tensor
                if isinstance(mask, torch.Tensor):
                    # If mask came back as HWC tensor from transform (less common but possible)
                    if mask.ndim == 3 and mask.shape[0] != 3 and mask.shape[2] == 3:
                        mask = mask.permute(2, 0, 1)  # HWC -> CHW
                    mask = mask.to(torch.uint8)
                else:  # If transform didn't return tensor (e.g., no ToTensorV2)
                    mask = mask.astype(np.uint8)  # Ensure correct type before tensor conversion later

                # Ensure image is float tensor
                if isinstance(img, torch.Tensor):
                    # If image came back as HWC tensor
                    if img.ndim == 3 and img.shape[0] != num_channels and img.shape[2] == num_channels:
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                    img = img.to(torch.float32)
                else:
                    img = img.astype(np.float32)  # Ensure correct type before tensor conversion later

            except Exception as e:
                print(f"Error during transform for {img_filename}: {e}. Using untransformed data.")
                # Re-stack original masks if transform failed midway and modified 'mask' var
                mask = np.stack([field_mask, edge_mask, contact_mask], axis=0)
                if self.scale_factor != 1.0:  # Re-apply resize if necessary
                    target_h, target_w = img_shape
                    resized_mask_channels = [
                        cv2.resize(mask[i], (target_w, target_h), interpolation=cv2.INTER_NEAREST) for i in range(3)
                    ]
                    mask = np.stack(resized_mask_channels, axis=0)
                mask = mask * 255
                mask = mask.astype(np.uint8)  # Ensure type
                # img is already normalized np array here

        # --- Convert to Tensors ---
        try:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img.copy()).float()  # Ensure CHW if numpy
            else:  # Already a tensor from transform
                img_tensor = img.float()  # Ensure type

            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask.copy()).to(torch.uint8)  # Ensure CHW if numpy
            else:  # Already a tensor from transform
                mask_tensor = mask.to(torch.uint8)  # Ensure type

        except Exception as e:
            print(f"Error converting data to tensors for {img_filename}: {e}")
            raise TypeError(f"Could not convert data to tensors for {img_filename}") from e

        # --- Final shape check ---
        if img_tensor.shape[1:] != mask_tensor.shape[1:]:
            # Add more detail to the error message
            raise ValueError(
                f"Final tensor shape mismatch for {img_filename}: "
                f"Image shape {img_tensor.shape}, Mask shape {mask_tensor.shape}. "
                f"This likely happened after transformations or resizing. "
                f"Check transform pipeline and interpolation methods."
            )
        return img_tensor, mask_tensor, self.img_filenames[idx]
