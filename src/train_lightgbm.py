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

# utils.dataset が正しいパスにあることを確認
from utils.dataset import FieldSegmentationDataset

import joblib
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import lightgbm as lgb
from PIL import Image

warnings.simplefilter("ignore")

# --- 定数 (変更なし) ---
IMAGE_DIR = "/workspace/projects/solafune-field-area-segmentation/data/train_images"
ANNOTATION_FILE = "/workspace/projects/solafune-field-area-segmentation/data/train_annotation.json"
CACHE_DIR = "/workspace/projects/solafune-field-area-segmentation/sandbox/outputs/cache"
SCALE_FACTOR = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PRED_DIR = "/workspace/projects/solafune-field-area-segmentation/outputs/ex8_with_aug" # 出力先変更
N_SPLITS = 4
RANDOM_STATE = 136
transform = None

os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)


train_path = "/workspace/projects/solafune-field-area-segmentation/data/train_images"
trains = glob.glob(f"{train_path}/*")
trains.sort()

if __name__ == "__main__":
    train_idxes = list(range(0, 45))

    print("--- Preparing Training Data ---")
    train_dataset = FieldSegmentationDataset(
        img_dir=IMAGE_DIR,
        ann_json_path=ANNOTATION_FILE,
        cache_dir=CACHE_DIR,
        scale_factor=SCALE_FACTOR,
        transform=None,
        contact_width=5,
        edge_width=3,
        img_idxes=train_idxes,
        mean=None,
        std=None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4, # 問題発生時は 0 を試す
        pin_memory=True if DEVICE == "cuda" else False,
        drop_last=False,
    )

    X = []
    y = []
    g = []
    aug_type = [] # ★ Augmentationの種類を記録するリストを追加 (0: original, 1: flip_v, 2: flip_h)
    img_meta = {}

    print("--- Extracting Features and Labels (With Augmentation) ---")
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        img_tensor, mask_tensor, file_name = data
        file_name_str = file_name[0] if isinstance(file_name, (list, tuple)) else file_name

        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        mask_np = mask_tensor.squeeze(0).permute(1, 2, 0).numpy()

        H, W, _ = img_np.shape
        img_meta[i] = {'shape': (H, W), 'filename': file_name_str}

        # --- 特徴量エンジニアリング (変更なし) ---
        NDVI = (img_np[:, :, 7] - img_np[:, :, 3]) / (img_np[:, :, 7] + img_np[:, :, 3] + 1e-8)
        img_np = np.concatenate((img_np, NDVI[:, :, np.newaxis]), axis=2)
        band_for_train_index = [1, 2, 3, 7, 12]
        img_features = img_np[:, :, band_for_train_index]
        mask_target = mask_np[:, :, 0]
        num_pixels = H * W

        # --- データ追加 (Augmentation 処理を復活) ---
        # 1. 元の画像
        X.append(img_features.reshape(-1, len(band_for_train_index)))
        y.append(mask_target.reshape(-1))
        g.append(np.full((num_pixels,), i))
        aug_type.append(np.zeros((num_pixels,), dtype=np.uint8)) # ★ aug_type: 0 (original)

        # 2. 上下反転
        img_flipped_v = cv2.flip(img_features, 0)
        mask_flipped_v = cv2.flip(mask_target, 0)
        X.append(img_flipped_v.reshape(-1, len(band_for_train_index)))
        y.append(mask_flipped_v.reshape(-1))
        g.append(np.full((num_pixels,), i))
        aug_type.append(np.ones((num_pixels,), dtype=np.uint8)) # ★ aug_type: 1 (flip_v)

        # 3. 左右反転
        img_flipped_h = cv2.flip(img_features, 1)
        mask_flipped_h = cv2.flip(mask_target, 1)
        X.append(img_flipped_h.reshape(-1, len(band_for_train_index)))
        y.append(mask_flipped_h.reshape(-1))
        g.append(np.full((num_pixels,), i))
        aug_type.append(np.full((num_pixels,), 2, dtype=np.uint8)) # ★ aug_type: 2 (flip_h)

    if not X:
        raise ValueError("No data loaded.")

    X = np.vstack(X)
    y = np.hstack(y)
    g = np.hstack(g)
    aug_type = np.hstack(aug_type) # ★ aug_type も hstack する

    print(f"Final Data shapes: X={X.shape}, y={y.shape}, g={g.shape}, aug_type={aug_type.shape}")
    print(f"Number of unique groups: {len(np.unique(g))}")
    print(f"Collected metadata for {len(img_meta)} images.")

    # --- Training ---
    # GroupKFold は shuffle 不要なことが多い
    gkfold = GroupKFold(n_splits=N_SPLITS)

    models = []

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 128, # メモリ考慮
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1, # 全コア使用に変更
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "verbose": -1,
    }

    # --- KFold ループ ---
    for i, (train_idx, valid_idx) in enumerate(gkfold.split(X, y, groups=g)):
        print(f"\n--- Fold {i+1}/{N_SPLITS} ---")
        train_x, train_y = X[train_idx], y[train_idx]
        val_x, val_y = X[valid_idx], y[valid_idx]
        val_g = g[valid_idx]
        val_aug_type = aug_type[valid_idx] # ★ バリデーションデータの Augmentation タイプを取得

        # グループリークチェック (変更なし)
        train_groups = np.unique(g[train_idx])
        valid_groups = np.unique(val_g)
        assert len(np.intersect1d(train_groups, valid_groups)) == 0, f"Fold {i}: Groups leaked!"
        print(f"Training on groups: {train_groups}")
        print(f"Validating on groups: {valid_groups}")

        # モデル学習 (変更なし)
        m = lgb.LGBMClassifier(**lgb_params)
        print(f"Training model Fold {i+1}...")
        m.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)],
        )
        models.append(m)

        # --- バリデーションデータに対する予測と保存 (Augmentation 対応) ---
        print(f"Predicting on validation set for Fold {i + 1}...")
        # クラス1の予測確率を取得
        val_pred_probas = m.predict_proba(val_x)[:, 1]
        # 閾値0.5で予測ラベルを決定
        val_pred_labels = (val_pred_probas > 0.5).astype(np.uint8)

        unique_val_groups = np.unique(val_g)
        print(f"Saving augmented predictions for {len(unique_val_groups)} validation images in Fold {i + 1}...")

        for group_id in tqdm(unique_val_groups, desc=f"Saving Fold {i + 1} Val Predictions"):
            try:
                meta = img_meta[group_id]
                H, W = meta["shape"]
                original_filename = meta["filename"]
                file_stem = os.path.splitext(original_filename)[0]
                expected_pixels = H * W
            except KeyError:
                print(f"Warning: Metadata for group {group_id} not found. Skipping.")
                continue
            except Exception as e:
                print(f"Error retrieving metadata for group {group_id}: {e}. Skipping.")
                continue

            # 現在のグループIDに対応するマスク
            group_mask_in_val = (val_g == group_id)

            # このグループの予測ラベルと Augmentation タイプを取得
            group_preds = val_pred_labels[group_mask_in_val]
            group_aug_types = val_aug_type[group_mask_in_val]

            # Augmentation タイプごとに予測を分離
            original_preds = group_preds[group_aug_types == 0]
            flipped_v_preds = group_preds[group_aug_types == 1]
            flipped_h_preds = group_preds[group_aug_types == 2]

            # --- 各予測の形状チェックと反転処理 ---
            pred_masks = []
            # 1. 元の予測
            if len(original_preds) == expected_pixels:
                pred_masks.append(original_preds.reshape(H, W))
            else:
                print(f"Warning (Fold {i+1}, Group {group_id}): Original prediction size mismatch. Expected {expected_pixels}, got {len(original_preds)}.")
                # サイズが違う場合はアンサンブルに含めないか、エラー処理をする
                continue # このグループの保存をスキップ

            # 2. 上下反転の予測 -> 元に戻す
            if len(flipped_v_preds) == expected_pixels:
                flipped_v_mask = flipped_v_preds.reshape(H, W)
                unflipped_v_mask = cv2.flip(flipped_v_mask, 0) # 上下反転を戻す
                pred_masks.append(unflipped_v_mask)
            else:
                print(f"Warning (Fold {i+1}, Group {group_id}): Flipped_v prediction size mismatch. Expected {expected_pixels}, got {len(flipped_v_preds)}.")
                continue

            # 3. 左右反転の予測 -> 元に戻す
            if len(flipped_h_preds) == expected_pixels:
                flipped_h_mask = flipped_h_preds.reshape(H, W)
                unflipped_h_mask = cv2.flip(flipped_h_mask, 1) # 左右反転を戻す
                pred_masks.append(unflipped_h_mask)
            else:
                print(f"Warning (Fold {i+1}, Group {group_id}): Flipped_h prediction size mismatch. Expected {expected_pixels}, got {len(flipped_h_preds)}.")
                continue

            # --- アンサンブル (多数決) ---
            # 3つのマスクが揃っている場合のみ実行
            if len(pred_masks) == 3:
                # uint8 のまま足し合わせる
                sum_mask = pred_masks[0] + pred_masks[1] + pred_masks[2]
                # 2つ以上が1であれば1とする (多数決)
                final_mask = (sum_mask >= 2).astype(np.uint8)

                # --- PNGで保存 ---
                pred_mask_img_save = (final_mask * 255).astype(np.uint8)
                output_filename = os.path.join(OUTPUT_PRED_DIR, f"fold{i + 1}_group{group_id}_{file_stem}_pred_aug_ensemble.png")
                try:
                    pil_img = Image.fromarray(pred_mask_img_save)
                    pil_img.save(output_filename)
                except Exception as e:
                    print(f"Error saving ensemble prediction PNG for group {group_id}, fold {i + 1}: {e}")
            else:
                 print(f"Skipping ensemble for group {group_id}, fold {i+1} due to missing predictions.")


        # モデル保存 (オプション)
        model_filename = f"lgb_model_fold{i+1}_with_aug.joblib"
        joblib.dump(m, model_filename)
        print(f"Model for Fold {i+1} saved as {model_filename}")

    print("\n--- Processing Finished ---")