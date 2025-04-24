# `train.py` への CosineAnnealingLR 導入計画

## 概要

`solafune-field-area-segmentation/src/train.py` を変更し、`torch.optim.lr_scheduler.CosineAnnealingLR` を使用して学習率を制御するようにします。

## パラメータ設定

*   `T_MAX`: 500
*   `ETA_MIN`: 1e-6

## 具体的な手順

1.  **対象ファイル:** `solafune-field-area-segmentation/src/train.py`
2.  **変更内容:**
    *   **インポート:** `torch.optim.lr_scheduler` から `CosineAnnealingLR` をインポートします。
    *   **設定パラメータ追加:** `if __name__ == "__main__":` ブロック内に以下のパラメータを追加します。
        *   `LR_SCHEDULER_T_MAX = 500`
        *   `LR_SCHEDULER_ETA_MIN = 1e-6`
    *   **`train_model` 関数シグネチャ変更:** 関数の引数に `lr_scheduler_t_max` と `lr_scheduler_eta_min` を追加します。
    *   **スケジューラ初期化:** `train_model` 関数内で、`optimizer` の初期化後に `CosineAnnealingLR` を初期化します。
        ```python
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_scheduler_t_max, eta_min=lr_scheduler_eta_min)
        ```
    *   **スケジューラ更新:** `train_model` 関数の各エポックの最後に `scheduler.step()` を呼び出します。
    *   **関数呼び出し変更:** `if __name__ == "__main__":` ブロックから `train_model` を呼び出す際に、追加したパラメータ (`LR_SCHEDULER_T_MAX`, `LR_SCHEDULER_ETA_MIN`) を渡します。
    *   **（オプション）ログ出力:** エポックごとのログに現在の学習率 (`optimizer.param_groups[0]['lr']`) を追加します。

## Mermaid ダイアグラムによるフロー

```mermaid
graph TD
    A[開始] --> B{ユーザー要望: CosineAnnealingLR 導入};
    B --> C{パラメータ確認: T_MAX=500, ETA_MIN=1e-6};
    C --> D{最終計画策定};
    D --> E[1. ライブラリインポート追加];
    D --> F[2. 設定パラメータ (T_MAX, ETA_MIN) 追加];
    D --> G[3. train_model関数変更: 引数追加];
    D --> H[4. train_model関数変更: スケジューラ初期化];
    D --> I[5. train_model関数変更: scheduler.step() 呼び出し];
    D --> J[6. train_model関数呼び出し変更];
    D --> K[7. (オプション) ログ出力変更];
    E & F & G & H & I & J & K --> L{ユーザーに最終計画提示};
    L --> M{計画承認};
    M --> N{計画をファイルに書き出し};
    N --> O[実装モードへ切り替え提案];
    O --> P[終了];
