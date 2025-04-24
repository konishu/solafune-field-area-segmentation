# WANDB Integration Plan for train.py

This document outlines the plan to integrate Weights & Biases (WANDB) logging into `solafune-field-area-segmentation/src/train.py`.

## Goal

Log training and validation metrics, hyperparameters, and validation image samples to WANDB during the training process defined in `train.py`.

## Information to Log

### Metrics (Per Epoch)

-   `avg_train_loss`
-   `avg_train_bce_loss`
-   `avg_train_dice_loss`
-   `current_lr` (logged as `learning_rate`)
-   `avg_val_loss`
-   `avg_val_bce_loss`
-   `avg_val_dice_loss`
-   `avg_val_dice_coeff` (Needs calculation during validation)

### Images (Per Validation Interval)

-   Concatenated images of (original image, ground truth mask, predicted mask) for the first few samples in the validation batch. Logged under the key `validation_samples`.

## WANDB Configuration

-   **API Key:** Read from `.env` file (`WANDB_API_KEY`). Assumes `python-dotenv` is installed manually by the user.
-   **Project Name:** `solafune-field-segmentation`
-   **Run Name:** Dynamically generated using the format `f"{EX_NUM}-{BACKBONE}"` (e.g., `ex2-maxvit_small_tf_512.in1k`).
-   **Config:** Log key hyperparameters defined in the `if __name__ == "__main__":` block (e.g., `BATCH_SIZE`, `NUM_EPOCHS`, `INITIAL_LR`, `BACKBONE`, etc.).

## Implementation Steps (`train.py`)

1.  **Import Libraries:**
    -   Add imports for `wandb`, `dotenv`, `torchvision`, `numpy`.
    ```python
    import wandb
    from dotenv import load_dotenv
    import torchvision
    import numpy as np
    # PIL might be needed if converting tensors to PIL Images explicitly,
    # but wandb.Image can often handle tensors directly.
    ```

2.  **Initialization (`if __name__ == "__main__":`)**
    -   Load environment variables early in the block:
        ```python
        load_dotenv()
        ```
    -   Define a dictionary containing hyperparameters to log:
        ```python
        # --- Configuration --- (Existing block)
        # ... define ROOT_DIR, EX_NUM, IMAGE_DIR, etc. ...

        # --- WANDB Config ---
        wandb_config = {
            "experiment_num": EX_NUM,
            "backbone": BACKBONE,
            "num_output_channels": NUM_OUTPUT_CHANNELS,
            "pretrained": PRETRAINED,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "initial_lr": INITIAL_LR,
            "warmup_epochs": WARMUP_EPOCHS,
            "min_lr": MIN_LR,
            "bce_weight": BCE_WEIGHT,
            "dice_weight": DICE_WEIGHT,
            "scale_factor": SCALE_FACTOR,
            "crop_h": CROP_H,
            "crop_w": CROP_W,
            "resize_h": RESIZE_H,
            "resize_w": RESIZE_W,
            "validation_interval": VALIDATION_INTERVAL,
            "dataset_mean": DATASET_MEAN, # Log mean/std if used
            "dataset_std": DATASET_STD,
            "optimizer": "AdamW", # Example: Log optimizer type
            "weight_decay": 1e-2, # Example: Log weight decay
            # Add other relevant parameters from the config section
        }
        run_name = f"{EX_NUM}-{BACKBONE}"
        # --- End WANDB Config ---
        ```
    -   Initialize WANDB run before the main `try...except` block or just inside it:
        ```python
        try:
            # --- Initialize WANDB ---
            wandb.init(
                project="solafune-field-segmentation",
                name=run_name,
                config=wandb_config
            )
            # --- End Initialize WANDB ---

            # ... existing setup code (datasets, dataloaders, model) ...

            # Start training
            train_model(
                # ... pass arguments ...
            )

            # ... save model ...

        except NameError as e:
            # ... existing error handling ...
        except FileNotFoundError as e:
            # ... existing error handling ...
        except Exception as e:
            # ... existing error handling ...
        finally:
            # --- Finish WANDB Run ---
            if wandb.run: # Check if wandb.init was successful and run is active
                print("Finishing WANDB run...")
                wandb.finish()
            # --- End Finish WANDB Run ---
            print("Training script finished.") # Keep or adjust final message
        ```

3.  **Modify `train_model` Function:**
    -   No need to pass `wandb_run` if using the global `wandb` object implicitly.
    -   **Inside Epoch Loop (End):** Log training metrics.
        ```python
        # After calculating avg_train_loss, avg_train_bce_loss, etc. and current_lr
        print(f"Epoch {epoch + 1} Avg Train Loss: ...") # Keep existing print
        wandb.log({
            "epoch": epoch + 1, # Log epoch number (1-based)
            "avg_train_loss": avg_train_loss,
            "avg_train_bce_loss": avg_train_bce_loss,
            "avg_train_dice_loss": avg_train_dice_loss,
            "learning_rate": current_lr,
        }) # Logged against the epoch step implicitly by wandb
        ```
    -   **Inside Validation Loop (End):**
        -   Calculate average Dice coefficient:
            ```python
            # Inside the validation interval check `if (epoch + 1) % validation_interval == 0:`
            # Before the validation loop starts
            running_val_dice_coeff = 0.0
            num_val_samples = 0

            # ... inside the validation batch loop `for val_batch in val_progress_bar:` ...
            # After calculating val_total_loss
            if not torch.isnan(val_total_loss): # Only calculate dice if loss is valid
                with torch.no_grad(): # Ensure no gradients are computed here
                    # Apply sigmoid to outputs for dice calculation
                    val_outputs_sigmoid = torch.sigmoid(val_outputs)
                    # Calculate dice coefficient using the existing dice_coeff function
                    # dice_coeff returns per-class dice: (N, C)
                    dice_coeffs_batch = dice_coeff(val_outputs_sigmoid, val_masks, smooth=1.0, epsilon=1e-6)
                    # Average over classes for each item in the batch -> (N,)
                    dice_per_item = dice_coeffs_batch.mean(dim=1)
                    # Sum the dice scores for the batch
                    running_val_dice_coeff += dice_per_item.sum().item()
                    num_val_samples += val_imgs.size(0) # Count samples processed

            # After the validation loop finishes
            # Avoid division by zero if num_val_samples is 0
            avg_val_dice = running_val_dice_coeff / num_val_samples if num_val_samples > 0 else 0.0
            print(f"Epoch {epoch + 1} Avg Valid Loss: ... Dice Coeff: {avg_val_dice:.4f}") # Add Dice to print
            ```
        -   Prepare log dictionary and log validation images:
            ```python
            # Still inside the validation interval check, after calculating avg metrics and avg_val_dice
            log_dict = {
                "epoch": epoch + 1, # Log against epoch
                "avg_val_loss": avg_val_loss,
                "avg_val_bce_loss": avg_val_bce_loss,
                "avg_val_dice_loss": avg_val_dice_loss,
                "avg_val_dice_coeff": avg_val_dice,
            }

            # --- Log Validation Images ---
            log_images = []
            num_samples_to_log = min(4, val_imgs.size(0)) # Log up to 4 samples from the *last* batch

            # Ensure tensors are on CPU and detached for processing
            val_imgs_cpu = val_imgs[:num_samples_to_log].cpu().detach()
            val_masks_cpu = val_masks[:num_samples_to_log].cpu().detach()
            val_outputs_cpu = val_outputs[:num_samples_to_log].cpu().detach()

            # Apply sigmoid and threshold for prediction mask visualization
            preds_sigmoid = torch.sigmoid(val_outputs_cpu)
            preds_binary = (preds_sigmoid > 0.5).float() # Threshold at 0.5

            for i in range(num_samples_to_log):
                # Input image: (C, H, W), assume C=3
                img = val_imgs_cpu[i]
                # Ground Truth Mask: (C, H, W), assume C=3 (field, edge, contact)
                # Prediction Mask: (C, H, W), same format
                # We can log each channel separately or combine them. Let's log combined.
                # Combine GT masks into one image (e.g., RGB)
                mask_gt_combined = val_masks_cpu[i] # Already (3, H, W)
                # Combine Pred masks into one image
                mask_pred_combined = preds_binary[i] # Already (3, H, W)

                # Normalize image if needed (ToTensorV2 usually scales 0-1)
                # Clamp values just in case to avoid warnings
                img = torch.clamp(img, 0, 1)
                mask_gt_combined = torch.clamp(mask_gt_combined, 0, 1)
                mask_pred_combined = torch.clamp(mask_pred_combined, 0, 1)

                # Create a grid: [Image | GT Mask | Pred Mask]
                combined_image = torchvision.utils.make_grid(
                    [img, mask_gt_combined, mask_pred_combined],
                    nrow=3, padding=2, normalize=False # Already in [0,1] range
                )
                log_images.append(wandb.Image(combined_image,
                                caption=f"Epoch {epoch+1} Sample {i} (Img | GT | Pred)"))

            if log_images: # Only add if images were generated
                log_dict["validation_samples"] = log_images
            # --- End Log Validation Images ---

            # Log all validation metrics and images for this epoch
            wandb.log(log_dict)
            ```

4.  **Helper Function `dice_coeff` (Review):**
    -   The existing `dice_coeff` function (lines 21-37) seems appropriate. It takes raw logits (`pred`), applies sigmoid internally, and calculates the Dice coefficient per class. The modification in step 3. uses this function correctly by passing the sigmoid-applied predictions.

## Next Steps

1.  Review this plan.
2.  If approved, request switching to `code` mode to implement these changes in `solafune-field-area-segmentation/src/train.py`.