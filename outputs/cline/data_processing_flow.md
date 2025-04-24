# データ処理フロー：衛星画像畑領域セグメンテーション

このドキュメントは、衛星画像から畑領域をセグメンテーションするためのトレーニングコードにおける、画像の読み込みからモデルへの入力までのデータ処理フローを説明します。処理は主に `src/utils/dataset.py` の `FieldSegmentationDataset` クラスと `src/train.py` で行われます。

## 1. データセットの初期化 (`FieldSegmentationDataset.__init__`)

1.  **入力:**
    *   画像ディレクトリパス (`img_dir`)
    *   アノテーションJSONファイルパス (`ann_json_path`)
    *   初期リサイズ係数 (`scale_factor`)
    *   エッジ/コンタクトマスク生成パラメータ (`edge_width`, `contact_width`)
    *   データ拡張パイプライン (`transform`)
    *   正規化用の平均/標準偏差 (`mean`, `std`)
2.  **処理:**
    *   アノテーションJSONを読み込み、画像ファイル名に対応するアノテーション情報を辞書 (`self.annotations`) に格納します。
    *   画像ディレクトリ内の `.tif` ファイルをリストアップし、アノテーションが存在するファイルのみを `self.img_filenames` に格納します。
    *   正規化用の `mean` と `std` を適切な形状 (C, 1, 1) に整形します。

## 2. データアイテムの取得 (`FieldSegmentationDataset.__getitem__`)

各データアイテム（画像と対応するマスク）を取得する際の処理フローです。

1.  **キャッシュの確認:**
    *   指定されたインデックス (`idx`) に対応する画像ファイル名を取得します。
    *   対応するキャッシュファイル (`.npz`) が存在するか確認します。
    *   **キャッシュが存在する場合:** キャッシュから前処理済みの画像 (`img`) とマスク (`mask`) を読み込み、ステップ 2-4 へ進みます。
    *   **キャッシュが存在しない場合:** ステップ 2-2 へ進みます。

2.  **画像読み込みと前処理 (キャッシュがない場合):**
    *   `rasterio` を使用して、対応する `.tif` 画像ファイルを読み込みます (12チャンネル、(C, H, W) 形式、`float32`)。
    *   `scale_factor` が 1.0 でない場合、`cv2.resize` を用いて画像を指定されたスケールにリサイズします (線形補間)。画像形状 (`img_shape`) も更新します。
    *   **正規化:**
        *   `mean` と `std` が提供されている場合、それらを用いて画像を正規化します: `(img - mean) / (std + epsilon)`。
        *   提供されていない場合、画像ごとに平均と標準偏差を計算し、正規化します。

3.  **マスク生成 (キャッシュがない場合):**
    *   画像に対応するアノテーションを取得します。
    *   空のラベルマップ (`labels`, (H, W), `uint16`) を初期化します。
    *   各アノテーション（畑領域）について:
        *   セグメンテーション情報（WKT または COCO 形式リスト）を解析します。
        *   `segmentation_to_mask` ヘルパー関数や `shapely` を用いて、ポリゴンからバイナリマスク (`poly_mask`) を生成します。`scale_factor` が適用されます。
        *   `poly_mask` が示す領域に、一意のインスタンスIDを `labels` マップに割り当てます。
    *   `labels` マップから以下の3つのマスクを生成します ((H, W), `uint8`):
        *   **`field_mask`:** `labels > 0` の領域。畑全体のマスク。
        *   **`edge_mask`:** 各インスタンスの境界。`skimage.morphology.erosion` を使用して各インスタンスマスクを縮小し、元のマスクとのXORを取ることで境界を抽出します。`edge_width` でカーネルサイズを指定します。
        *   **`contact_mask`:** インスタンス間の接触領域。`skimage.morphology.dilation` と `skimage.segmentation.watershed` を利用します。
            1.  `field_mask` を `contact_width` で指定されたカーネルで膨張 (`dilation`) させます (`dilated_field`)。
            2.  `labels` をマーカーとして `watershed` を実行し、境界線 (`watershed_lines`) を得ます。
            3.  `watershed_lines` と `edge_mask` を結合し、さらに `dilation` して接触候補領域 (`contact_mask_candidates`) を得ます。
            4.  候補領域の各ピクセルについて、元の `labels` マップにおける近傍に複数の異なるインスタンスIDが存在するか確認し、存在すれば接触点として `contact_mask` にマークします。
    *   生成された3つのマスク (`field_mask`, `edge_mask`, `contact_mask`) をチャンネル方向にスタックします (`mask`, (3, H, W), `uint8`)。
    *   **キャッシュ保存:** 生成された画像 (`img`) とマスク (`mask`) を `.npz` ファイルとしてキャッシュディレクトリに保存します。

4.  **後処理と変換:**
    *   `scale_factor` が 1.0 でない場合、スタックされた `mask` も画像と同じターゲットサイズにリサイズします (`cv2.resize`、最近傍補間)。
    *   マスクの値を 0-1 から 0-255 にスケーリングします (`mask = mask * 255`)。
    *   **データ拡張 (`self.transform`):** `train.py` で定義された Albumentations パイプラインが適用されます。
        *   画像とマスクを HWC 形式に変換します。
        *   `RandomCrop`: 指定されたサイズ (`CROP_H`, `CROP_W`) でランダムにクロップします。
        *   `Resize`: モデルの入力サイズ (`RESIZE_H`, `RESIZE_W`) にリサイズします (最近傍補間)。
        *   `HorizontalFlip`, `VerticalFlip`: ランダムに水平・垂直反転します。
        *   `ToTensorV2`:
            *   画像を CHW 形式の `torch.float32` テンソルに変換します (値は 0.0-1.0 にスケーリングされません。正規化は Dataset 内で実施済み)。
            *   マスクを CHW 形式の `torch.uint8` テンソルに変換します (値は 0 または 255)。
    *   変換後の画像テンソル (`img_tensor`) とマスクテンソル (`mask_tensor`) を返します。

## 3. データローダー (`DataLoader` in `train.py`)

1.  `FieldSegmentationDataset` インスタンスと設定（バッチサイズ `BATCH_SIZE`、ワーカー数 `NUM_WORKERS` など）を用いて `DataLoader` を初期化します。
2.  トレーニングループ内で、`DataLoader` は `FieldSegmentationDataset` から `__getitem__` を通じてデータアイテム（`img_tensor`, `mask_tensor`）を取得し、指定された `BATCH_SIZE` のミニバッチを生成します。

## 4. トレーニングループ (`train_model` in `train.py`)

1.  `DataLoader` からミニバッチを取得します (`imgs`, `masks`)。
2.  画像テンソル (`imgs`) とマスクテンソル (`masks`) を計算デバイス (`cuda` or `cpu`) に転送します。
3.  **マスクの前処理:** マスクテンソル (`masks`) を `torch.float32` 型に変換し、値を 255.0 で割って 0.0-1.0 の範囲にスケーリングします。これは、`BCEWithLogitsLoss` や `dice_loss` 関数がターゲットとして float 型のマスク（通常は 0.0 または 1.0）を期待するためです。
4.  画像テンソル (`imgs`) をモデルに入力し、予測結果 (`outputs`, raw logits) を得ます。
5.  予測結果 (`outputs`) と前処理されたマスク (`masks`) を用いて損失（BCE Loss + Dice Loss）を計算します。

## Mermaid図

```mermaid
graph TD
    subgraph Dataset Initialization (train.py)
        A[Input: img_dir, ann_json, params] --> B{Initialize FieldSegmentationDataset};
    end

    subgraph Get Item (dataset.py: __getitem__)
        C[DataLoader requests item idx] --> D{Check Cache?};
        D -- Yes --> E[Load img, mask from cache (.npz)];
        D -- No --> F[Load Image (.tif)];
        F --> G{Scale Image? (scale_factor)};
        G -- Yes --> H[Resize Image (cv2.resize)];
        G -- No --> I[Use Original Size];
        H --> I;
        I --> J[Normalize Image (mean/std)];
        J --> K[Generate Instance Label Map (labels) from Annotations];
        K --> L[Generate field_mask];
        K --> M[Generate edge_mask (erosion)];
        K --> N[Generate contact_mask (watershed, dilation)];
        L & M & N --> O[Stack Masks (3, H, W)];
        J & O --> P[Save img, mask to cache (.npz)];
        P --> E;
        E --> Q{Scale Mask? (scale_factor)};
        Q -- Yes --> R[Resize Mask (cv2.resize, NEAREST)];
        Q -- No --> S[Use Original Size Mask];
        R --> S;
        S --> T[Scale Mask Values (0-255)];
    end

    subgraph Transformations (train.py -> dataset.py)
        T --> U[Apply Albumentations Transform];
        U -- RandomCrop --> V[Crop];
        V -- Resize --> W[Resize to Model Input];
        W -- Flips --> X[Augment];
        X -- ToTensorV2 --> Y[Convert to Tensors (img: float, mask: uint8)];
    end

    subgraph DataLoader (train.py)
       Y --> Z[DataLoader yields Batch];
    end

    subgraph Training Loop (train.py)
        Z --> AA[Get Batch (imgs, masks)];
        AA --> BB[Move to Device];
        BB --> CC[Preprocess Mask (to float, scale 0-1)];
        CC --> DD[Input imgs to Model];
        DD --> EE[Get Predictions (outputs)];
        EE & CC --> FF[Calculate Loss (BCE + Dice)];
    end

    B --> C;