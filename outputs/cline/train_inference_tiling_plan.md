# src/train_inference.py 改修計画: タイリング推論の実装 (v2: サイズ不一致修正)

## 目的

`src/train_inference.py` を改修し、画像全体に対するセグメンテーション推論結果を、元の画像解像度（`scale_factor` 適用後）で得られるようにする。タイリング推論を導入し、オーバーラップする領域の予測値を平均化することで、滑らかな予測マップを生成する。

**問題点(v1からの変更理由):** v1の実装では、推論前に画像全体がモデル入力サイズ(1024x1024)にリサイズされていたため、最終出力も1024x1024になっていた。これを修正し、元の解像度で処理を行うように変更する。

## 依存関係

- `numpy`
- `opencv-python` (`cv2`)
- `torch`
- `albumentations`

タイリング処理に特化したライブラリは使用せず、`numpy` と `cv2` を用いて実装する。

## 改修計画 (v2)

### 1. データロード処理の変更

-   **`transform` の変更**:
    -   `src/train_inference.py` 内の `transform` 定義から `A.RandomCrop` を削除する。
    -   **`A.Resize` も削除する。** データローダーからは `scale_factor` 適用後の解像度の画像テンソルが出力されるようにする。`ToTensorV2()` は維持する。
-   **`DataLoader` の変更**:
    -   `batch_size` を `1` に変更する。
    -   `shuffle=False` は維持する。

### 2. 推論ループの変更

-   現在のデータローダーからのバッチ処理ループを、画像一枚ずつ処理するループに変更する。
-   **タイリング関数の実装**:
    -   ヘルパー関数 `create_tiles(image, tile_size_h, tile_size_w, stride_h, stride_w)` を実装する。
        -   `image`: 入力画像 (Tensor, `scale_factor` 適用後の解像度)
        -   `tile_size_h`: タイルの高さ (512)
        -   `tile_size_w`: タイルの幅 (512)
        -   `stride_h`: 縦方向のストライド (128)
        -   `stride_w`: 横方向のストライド (128)
    -   この関数は、分割されたタイルのリストと、各タイルが元の画像上のどの位置に対応するかを示す座標 (左上隅) のリストを返す。
    -   画像の端でタイルサイズに満たない部分はパディングして `(tile_size_h, tile_size_w)` のサイズにする。
-   **タイルごとの推論**:
    -   `create_tiles` で生成された各タイル (`tile_tensor`, サイズ: `(C, TILE_H, TILE_W)`) に対して以下を実行:
        -   **タイルリサイズ (入力前)**: タイルをモデルが期待する入力サイズ (`RESIZE_H`, `RESIZE_W` = 1024x1024) にリサイズする (`cv2.resize` または `torch.nn.functional.interpolate`)。リサイズ後のテンソルを `resized_tile_tensor` とする。
        -   `resized_tile_tensor` をモデル (`model`) に入力し、推論を実行する。出力 `tile_output` のサイズは `(NUM_OUTPUT_CHANNELS, RESIZE_H, RESIZE_W)` となる。
        -   `tile_output` に `torch.sigmoid` を適用する。
        -   **タイルリサイズバック (出力後)**: Sigmoid適用後の `tile_output` を、元のタイルのサイズ (`TILE_H`, `TILE_W` = 512x512) にリサイズし直す (`cv2.resize` または `torch.nn.functional.interpolate`)。リサイズバック後のテンソルを `resized_back_output` とする。
-   **結果のマージ**:
    -   データローダーから取得した画像テンソル (`img_tensor`) の高さ `original_h` と幅 `original_w` を取得する (これは `scale_factor` 適用後の解像度)。
    -   この `(original_h, original_w)` と同じサイズのテンソル (`full_prediction_map`) を用意し、ゼロで初期化する (データ型: `float32`)。
    -   同様にカウント用テンソル (`count_map`) も `(original_h, original_w)` サイズで用意し、ゼロで初期化する (データ型: `float32`)。
    -   各タイルの **リサイズバックされた** 推論結果 (`resized_back_output`) を、対応する座標に基づいて `full_prediction_map` の該当領域に加算する。**注意:** パディングされた領域は加算しないように、元のタイル領域 (`h_tile`, `w_tile`) のみを加算する。
    -   同時に、該当領域の `count_map` を `1` だけインクリメントする。
    -   全てのタイルの処理が終わったら、`full_prediction_map` を `count_map` で要素ごとに割ることで、オーバーラップ領域の予測値を平均化する (`averaged_prediction_map`)。ゼロ除算を避ける処理を行う。
-   **最終マスク生成**:
    -   平均化された確率マップ (`averaged_prediction_map`) に対して、既存の閾値処理 (`thresholds`) を適用し、最終的なセグメンテーションマスク (例: `final_mask`, uint8, 0 or 1) を生成する。このマスクのサイズは `(NUM_OUTPUT_CHANNELS, original_h, original_w)` となる。

### 3. 結果の保存

-   生成された画像全体のセグメンテーションマスク (`final_mask`) を `PREDICTION_DIR` に保存する。サイズは `(original_h, original_w)`。
-   ファイル名は、元の画像ファイル名に基づいたものにする (例: `original_image_pred_combined.png`)。
-   保存形式は、既存の処理に合わせて `cv2.imwrite` を使用し、値を 0-255 の範囲にスケーリングする。
-   クラスごとのマスク保存も同様に行う。
-   不要なリサイズ処理は削除済みであることを確認。

## 処理フローの概要 (Mermaid図 v2)

```mermaid
graph TD
    A[推論スクリプト開始] --> B(画像全体をロード・前処理 (scale_factor適用後解像度));
    B -- img_tensor (H, W) --> C{タイリング関数呼び出し (create_tiles, tile=512x512, stride=128x128)};
    C -- タイルリスト, 座標リスト --> D(予測マップとカウントマップを初期化 (H, W));
    D --> E{タイルごとにループ};
    E -- 各タイル (512x512) --> F[タイルをリサイズ (1024x1024)];
    F -- リサイズ後タイル --> G[モデルで推論実行];
    G -- 推論結果 (1024x1024) --> H[結果をリサイズバック (512x512)];
    H -- リサイズバック後結果 --> I[予測マップに結果を加算, カウントマップを更新];
    I --> E;
    E -- ループ終了 --> J[予測マップをカウントマップで割り平均化];
    J -- 平均化マップ (H, W) --> K[閾値処理を適用];
    K -- 最終マスク (H, W) --> L[画像全体のマスクを保存];
    L --> M[終了];
```

## 次のステップ

この更新された計画に基づき、`src/train_inference.py` のコード改修を再度行う。