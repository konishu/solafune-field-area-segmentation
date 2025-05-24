import os
import shutil
import sys
import json  # アノテーションファイル読み込みのため追加
import argparse  # YAML読み込みのため追加
import yaml  # YAML読み込みのため追加
import numpy as np
import torch
import torchvision  # For image logging
import wandb  # Import WANDB
from dotenv import load_dotenv  # Import dotenv
import gc  # For garbage collection

import albumentations as A
import cv2  # Import OpenCV
from albumentations.pytorch import ToTensorV2
from models.unet_maxvit import UNet
from torch import nn, optim
from torch.optim import lr_scheduler  # Import LR scheduler (LinearLR, CosineAnnealingLR, SequentialLR)
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold 

from utils.dataset import FieldSegmentationDataset
from utils.calc import dice_coeff, dice_loss

from train import train_model  # Import train_model function
from test_inference import predict_on_test_data  # Import predict function


def main():
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
    CACHE_DIR = os.path.join(ROOT_DIR, CACHE_DIR_BASE)
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
    ACCUMULATION_STEPS = cfg["training"].get("accumulation_steps", 1)
    EARLY_STOPPING_THRESHOLD = cfg["training"].get("early_stopping_threshold", 10)

    WANDB_PROJECT = cfg["wandb"]["project"]
    WANDB_LOG_IMAGES = cfg["wandb"]["log_images"]
    WANDB_LOG_IMAGE_FREQ = cfg["wandb"].get("log_image_freq", VALIDATION_INTERVAL)
    WANDB_NUM_IMAGES_TO_LOG = cfg["wandb"].get("num_images_to_log", 4)
    run_name = f"{EX_NUM}-{BACKBONE}"

    TILE_H = cfg["test"].get("tile_h", 512)  # Use crop size if tile size not specified
    TILE_W = cfg["test"].get("tile_w", 512)  # Use crop size if tile size not specified
    STRIDE_H = cfg["test"].get("stride_h", 256)  # Default stride
    STRIDE_W = cfg["test"].get("stride_w", 256)  # Default stride
    PREDECT_DIR = os.path.join(cfg["test"].get("predicted_mask_dir", "predicted_masks"))
    TEST_IMG_DIR = os.path.join(ROOT_DIR, cfg["test"].get("img_dir", "data/test_images"))
    TEST_CLASS_THRESHOLDS = cfg["test"].get("class_thresholds", [0.5] * NUM_OUTPUT_CHANNELS)

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
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_img_idxed = list(range(len(all_img_filenames)))
    for fold, (train_idxes, valid_idxes) in enumerate(kf.split(train_img_idxed)):
        print(f"Fold {fold + 1}: Train indices {train_idxes}, Validation indices {valid_idxes}")

        try:
            wandb_config_log = {
                "config_file": args.config,
                "experiment": cfg["experiment"],
                "data": cfg["data"],
                "model": cfg["model"],
                "training": cfg["training"],
                "optimizer": "AdamW",
            }
            wandb.init(project=WANDB_PROJECT, name=run_name + f'_{fold}', config=wandb_config_log)

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
            if VALID_IMG_INDEX:
                print("Initializing Validation Dataset...")
                valid_dataset = FieldSegmentationDataset(
                    img_dir=IMAGE_DIR,
                    ann_json_path=ANNOTATION_FILE,
                    cache_dir=CACHE_DIR,
                    scale_factor=SCALE_FACTOR,
                    transform=transform_valid,
                    contact_width=CONTACT_WIDTH,
                    edge_width=EDGE_WIDTH,
                    img_idxes=valid_idxes,
                    mean=DATASET_MEAN,
                    std=DATASET_STD,
                )

            print(f"{train_idxes=}")
            print(f"{valid_idxes=}")

            if len(train_dataset) == 0:
                print("Error: Training dataset is empty after filtering. Check file paths and validation indices.")
                sys.exit()
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
                    batch_size=1,
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

            final_best_dice_checkpoint_path = train_model(
                model=model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                validation_interval=VALIDATION_INTERVAL,
                output_dir=OUTPUT_DIR,
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
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )

            model_save_path = os.path.join(ROOT_DIR, OUTPUT_DIR_BASE, EX_NUM)
            os.makedirs(model_save_path, exist_ok=True)
            final_model_name = "model_final.pth"
            torch.save(model.state_dict(), os.path.join(model_save_path, final_model_name))
            print(f"Model saved to {os.path.join(model_save_path, final_model_name)}")
            
            del model, train_dataset, valid_dataset, train_dataloader, valid_dataloader
            gc.collect()  # 明示的にガベージコレクションを試みる
            torch.cuda.empty_cache()  # Clear GPU memory
            print("Model and datasets cleared from memory.")

            ###############################
            # test_inference.pyの実行
            ###############################
            model = UNet(backbone_name=BACKBONE, pretrained=PRETRAINED, num_classes=NUM_OUTPUT_CHANNELS, img_size=RESIZE_W)
            model.load_state_dict(torch.load(final_best_dice_checkpoint_path))
            model.to(DEVICE)
            model.eval()
            print(f"Model loaded from {final_best_dice_checkpoint_path}")
            print("Setting up test dataset...")
            print("Running inference on test images...")
            os.makedirs(PREDECT_DIR, exist_ok=True)

            test_dataset = FieldSegmentationDataset(
                img_dir=TEST_IMG_DIR,
                scale_factor=SCALE_FACTOR,
                transform=A.Compose([ToTensorV2()]),
                contact_width=CONTACT_WIDTH,
                edge_width=EDGE_WIDTH,
                cache_dir=CACHE_DIR,
                is_test_mode=True,
            )
            
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )
            
            test_result_saved_path = os.path.join(PREDECT_DIR, f"{BACKBONE}_scalefactor_{SCALE_FACTOR}", str(fold))
            os.makedirs(test_result_saved_path, exist_ok=True)
            predict_on_test_data(
                model=model,
                dataset=test_dataset,
                dataloader=test_dataloader,
                device=DEVICE,
                class_thresholds=TEST_CLASS_THRESHOLDS,
                num_output_channels=NUM_OUTPUT_CHANNELS,
                tile_h=TILE_H,
                tile_w=TILE_W,
                stride_h=STRIDE_H,
                stride_w=STRIDE_W,
                resize_h=RESIZE_H,
                resize_w=RESIZE_W,
                prediction_dir=test_result_saved_path,
            )
            
            del model, test_dataset, test_dataloader
            torch.cuda.empty_cache()  # Clear GPU memory after inference
            print("Inference completed and results saved.")

            # if final_best_dice_checkpoint_path and os.path.exists(final_best_dice_checkpoint_path):
            #     # コピーして固定名で保存 (例: model_best_dice.pth)
            #     destination_best_dice_path = os.path.join(OUTPUT_DIR, "model_best_dice.pth")
            #     shutil.copyfile(final_best_dice_checkpoint_path, destination_best_dice_path)
            #     print(
            #         f"Best dice model (from {final_best_dice_checkpoint_path}) also saved as {destination_best_dice_path}"
            #     )
            #     if wandb.run:  # WandBにも最終的なベストパスを記録 (サマリーではなくアーティファクトの方が良いかもしれない)
            #         wandb.summary["final_selected_best_dice_model_path_on_disk"] = destination_best_dice_path
            # else:
            #     print("No best dice checkpoint was saved during training, or the path was invalid.")

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


if __name__ == "__main__":
    main()
