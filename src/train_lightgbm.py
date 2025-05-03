import glob
import tifffile
import numpy as np
import lightgbm as lgb
import warnings
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon, mapping

import json
import os
from collections import defaultdict
import rasterio
from torch.utils.data import DataLoader

from dataset import FieldSegmentationDataset

import joblib
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import lightgbm as lgb

warnings.simplefilter("ignore")

IMAGE_DIR = "/workspace/projects/solafune-field-area-segmentation/data/train_images"
ANNOTATION_FILE = "/workspace/projects/solafune-field-area-segmentation/data/train_annotation.json"
CACHE_DIR = "/workspace/projects/solafune-field-area-segmentation/sandbox/outputs/cache"
SCALE_FACTOR = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = None


train_path = "/workspace/projects/solafune-field-area-segmentation/data/train_images"
trains = glob.glob(f"{train_path}/*")
trains.sort()

if __name__ == "__main__":
    train_idxes = list(range(0, 45))
    valid_idxes = list(range(45, 50))

    train_dataset = FieldSegmentationDataset(
        img_dir=IMAGE_DIR,
        ann_json_path=ANNOTATION_FILE,
        cache_dir=CACHE_DIR,
        scale_factor=SCALE_FACTOR,
        transform=None,  # Use training transforms
        contact_width=5,  # Use cfg value
        edge_width=3,  # Use cfg value
        img_idxes=train_idxes,
        mean=None,
        std=None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,  # Shuffle training data
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
        drop_last=True,  # Consider dropping last incomplete batch
    )

    X = []
    y = []
    g = []

    mask_own = defaultdict(str)
    for i, data in tqdm(enumerate(train_dataloader)):
        img_tensor, mask_tensor, file_name = data

        # (C, H, W) -> (H, W, C)
        img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 12)
        mask_tensor = mask_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

        H, W, _ = img_tensor.shape

        # NDVIの特徴量を追加
        NDVI = (img_tensor[:, :, 7] - img_tensor[:, :, 3]) / (img_tensor[:, :, 7] + img_tensor[:, :, 3] + 1e-8)
        # NDVIの特徴量をimg_tensorに追加
        img_tensor = np.concatenate((img_tensor, NDVI[:, :, np.newaxis]), axis=2)  # (H, W, 13)

        # img_tensorについて、Band2,3,4,8,13を抽出
        band_for_train_index = [1, 2, 3, 7, 12]
        img_tensor = img_tensor[:, :, band_for_train_index]  # (H, W, 5)

        ### maskのうち0チャンネルだけを抽出
        mask_tensor = mask_tensor[:, :, 0]

        ## データ追加
        print(f"{img_tensor.shape=}, {mask_tensor.shape=}")
        X.append(img_tensor.reshape(-1, len(band_for_train_index)))  # (H*W, 5)
        y.append(mask_tensor.reshape(-1))
        g.append(np.full((H * W,), i))  # 画像単位でグループ番号

        ## 上下に反転
        img_tensor_ = cv2.flip(img_tensor, 0)
        mask_tensor_ = cv2.flip(mask_tensor, 0)
        X.append(img_tensor_.reshape(-1, len(band_for_train_index)))
        y.append(mask_tensor_.reshape(-1))
        g.append(np.full((H * W,), i))
        ## 左右に反転
        img_tensor_ = cv2.flip(img_tensor, 1)
        mask_tensor_ = cv2.flip(mask_tensor, 1)
        X.append(img_tensor_.reshape(-1, len(band_for_train_index)))
        y.append(mask_tensor_.reshape(-1))
        g.append(np.full((H * W,), i))

    X = np.vstack(X)
    y = np.hstack(y)
    g = np.hstack(g)
    print(f"{X.shape=}, {y.shape=}, {g.shape=}")

    # Training
    gkfold = GroupKFold(n_splits=4, shuffle=True, random_state=136)

    models = []

    lgb_params = {
        "boosting_type": "gbdt",
        "num_leaves": 256,
        "max_depth": 75,
        "n_estimators": 5000,
        "random_state": 136,
        "verbose": -1,
    }

    for i, (train_idx, valid_idx) in enumerate(gkfold.split(X, y, groups=g)):
        train_x = X[train_idx]
        train_y = y[train_idx]

        val_x = X[valid_idx]
        val_y = y[valid_idx]

        m = lgb.LGBMClassifier(**lgb_params)
        print(f"Training model {i}...")
        m.fit(
            train_x,
            train_y,
            eval_metric="logloss",
            eval_set=[(val_x, val_y)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
        )
        models.append(m)

        # モデルの重みを保存
        model_filename = f"lgb_model_{i}.pkl"
        joblib.dump(m, model_filename)
        print(f"Model {i} saved as {model_filename}")

    # Prediction
    val_dataset = FieldSegmentationDataset(
        img_dir=IMAGE_DIR,
        ann_json_path=ANNOTATION_FILE,
        cache_dir=CACHE_DIR,
        scale_factor=SCALE_FACTOR,
        transform=None,  # Use training transforms
        contact_width=5,  # Use cfg value
        edge_width=3,  # Use cfg value
        img_idxes=[0, 1, 2, 47, 48, 49],  # Use calculated train indices
        mean=None,
        std=None,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,  # Shuffle training data
        num_workers=1,
        pin_memory=True if DEVICE == "cuda" else False,
        drop_last=True,  # Consider dropping last incomplete batch
    )
