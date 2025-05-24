import math
import torch
import yaml

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