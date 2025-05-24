import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils.helper_func import create_tiles


# テストデータに対する予測関数
def predict_on_test_data(
    model,
    dataset,
    dataloader,
    device,
    class_thresholds=[0.3, 0.1, 0.1],
    num_output_channels=3,
    tile_h=512,
    tile_w=512,
    stride_h=256,
    stride_w=256,
    resize_h=512,
    resize_w=512,
    prediction_dir="predictions",
):
    model.eval()

    with torch.no_grad():
        progress_bar_infer = tqdm(enumerate(dataloader), total=len(dataloader), desc="Inferring Images")
        for idx, batch in progress_bar_infer:
            if batch is None:
                print(f"Warning: Skipping empty batch at index {idx}")
                continue

            img_tensor, _, _ = batch  # Get the single image tensor (C, H, W)
            img_tensor = img_tensor.squeeze(0)  # Remove batch dimension -> (C, H, W)
            c, original_h, original_w = img_tensor.shape  # Get original dimensions *after* transform (Resize)

            full_prediction_map = torch.zeros(
                (num_output_channels, original_h, original_w), dtype=torch.float32, device=device
            )
            count_map = torch.zeros((original_h, original_w), dtype=torch.float32, device=device)

            tiles, coords = create_tiles(img_tensor, tile_h, tile_w, stride_h, stride_w)
            print(f"Image {idx}: Created {len(tiles)} tiles.")

            tile_progress_bar = tqdm(zip(tiles, coords), total=len(tiles), desc=f"  Tiles Img {idx}", leave=False)
            for tile_data, (y_start, x_start) in tile_progress_bar:
                tile_tensor = tile_data.to(device)  # Tile is already (C, TILE_H, TILE_W)
                tile_tensor_batch = tile_tensor.unsqueeze(0)  # (1, C, TILE_H, TILE_W)
                resized_tile_tensor = torch.nn.functional.interpolate(
                    tile_tensor_batch, size=(resize_h, resize_w), mode="bilinear", align_corners=False
                )  # (1, C, RESIZE_H, RESIZE_W)

                tile_output = model(resized_tile_tensor)
                tile_output = torch.sigmoid(tile_output)
                resized_back_output = torch.nn.functional.interpolate(
                    tile_output, size=(tile_h, tile_w), mode="bilinear", align_corners=False
                ).squeeze(0)
                y_end = min(y_start + tile_h, original_h)
                x_end = min(x_start + tile_w, original_w)
                h_tile, w_tile = (
                    y_end - y_start,
                    x_end - x_start,
                )
                full_prediction_map[:, y_start:y_end, x_start:x_end] += resized_back_output[:, :h_tile, :w_tile]
                count_map[y_start:y_end, x_start:x_end] += 1
            count_map[count_map == 0] = 1e-6
            averaged_prediction_map = full_prediction_map / count_map.unsqueeze(0)
            final_mask = torch.zeros_like(averaged_prediction_map, dtype=torch.uint8)
            for class_idx, threshold in enumerate(class_thresholds):
                final_mask[class_idx, :, :] = (averaged_prediction_map[class_idx, :, :] > threshold).byte()
            if idx < len(dataset.img_filenames):
                original_img_filename = dataset.img_filenames[idx]
                output_filename_base = os.path.splitext(original_img_filename)[0] + "_pred_tiled.png"
                final_mask_np = final_mask.cpu().numpy()
                final_mask_np = np.transpose(final_mask_np, (1, 2, 0))
                output_h, output_w = final_mask_np.shape[:2]
                if output_h != original_h or output_w != original_w:
                    print(f"Warning: Size mismatch for {original_img_filename}!")
                    print(f"  Input size (after scale_factor): ({original_h}, {original_w})")
                    print(f"  Output mask size: ({output_h}, {output_w})")
                else:
                    print(f"Output mask size ({output_h}, {output_w}) matches input size for {original_img_filename}.")
                try:
                    for class_idx in range(final_mask_np.shape[2]):
                        class_output_path = os.path.join(
                            prediction_dir, f"{os.path.splitext(output_filename_base)[0]}_class_{class_idx}.png"
                        )
                        mask_to_save = (final_mask_np[:, :, class_idx] * 255).astype(np.uint8)
                        cv2.imwrite(class_output_path, mask_to_save)

                    combined_mask_bgr = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                    if final_mask_np.shape[2] > 0:
                        combined_mask_bgr[:, :, 0] = (final_mask_np[:, :, 0] * 255).astype(np.uint8)  # Blue
                    if final_mask_np.shape[2] > 1:
                        combined_mask_bgr[:, :, 1] = (final_mask_np[:, :, 1] * 255).astype(np.uint8)  # Green
                    if final_mask_np.shape[2] > 2:  # noqa: PLR2004
                        combined_mask_bgr[:, :, 2] = (final_mask_np[:, :, 2] * 255).astype(np.uint8)  # Red

                    combined_output_path = os.path.join(
                        prediction_dir, f"{os.path.splitext(output_filename_base)[0]}_combined.png"
                    )
                    cv2.imwrite(combined_output_path, combined_mask_bgr)
                    print(f"Output mask saved for {original_img_filename} to {combined_output_path}")
                    progress_bar_infer.set_postfix(saved=original_img_filename)
                except (OSError, TypeError, cv2.error) as e:
                    print(f"Error saving output mask for {original_img_filename}: {e}")
            else:
                print(f"Warning: Index {idx} out of bounds for dataset filenames.")
        print(f"\nTiling Inference complete. Predictions saved to {prediction_dir}")
        if idx >= len(dataset.img_filenames):
            print(f"Warning: Index {idx} out of bounds for dataset filenames.")
