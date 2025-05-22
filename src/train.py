import os
import torch
import torchvision
import wandb

from torch import optim
from torch.optim import lr_scheduler  # Import LR scheduler (LinearLR, CosineAnnealingLR, SequentialLR)
from tqdm import tqdm

from utils.calc import dice_coeff, dice_loss


def train_model(
    model,
    train_dataloader,  # Renamed from dataloader
    valid_dataloader,  # Added for validation
    validation_interval,  # Added for validation frequency
    output_dir=None,
    num_epochs=500,
    device="cuda",
    bce_weight=0.5,
    dice_weight=0.5,
    initial_lr=4e-4,  # Add initial_lr parameter
    warmup_epochs=10,  # Add warmup_epochs parameter
    cosine_decay_epochs=500,  # Note: This is effectively num_epochs in the scheduler logic below
    min_lr=1e-6,
    # lr_scheduler_t_max and lr_scheduler_eta_min are effectively num_epochs and min_lr for CosineAnnealingLR
    weight_decay=1e-2,  # Default value if not passed from main
    wandb_log_images=True,  # Default value if not passed from main
    wandb_num_images_to_log=4,  # Default value if not passed from main
    accumulation_steps=1,  # <<< Minimal change: Added argument with default
    pos_weight_ratio=[0.11, 99.0, 19.0],  # Default pos_weight for BCE loss
    early_stopping_threshold=10,  # Early stopping threshold
):
    """
    Trains the U-Net model using BCE + Dice loss with Linear Warmup + Cosine Decay LR schedule,
    and performs validation periodically.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, switching to CPU.")
        device = "cpu"
    model.to(device)

    # Use BCEWithLogitsLoss (reduction='mean' is simpler here)
    pos_weight = torch.tensor(
        pos_weight_ratio,
        device=device,
        dtype=torch.float,
    ).view(1, -1, 1, 1)
    criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion_bce = nn.BCEWithLogitsLoss()  # Default reduction='mean'
    # No need for separate Dice criterion instance if using the function directly

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    actual_cosine_epochs = num_epochs - warmup_epochs
    if actual_cosine_epochs <= 0:
        print(f"Warning: warmup_epochs ({warmup_epochs}) >= num_epochs ({num_epochs}). Cosine decay part will not run.")
        actual_cosine_epochs = 1

    if warmup_epochs >= num_epochs:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_epochs)
    elif warmup_epochs > 0:
        scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_cosine_epochs, eta_min=min_lr)
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    best_dice_score_for_checkpointing = -1.0
    best_dice_checkpoint_epoch = -1
    best_dice_checkpoint_path = None

    print(
        f"Starting training on {device} for {num_epochs} epochs (BCE weight: {bce_weight}, Dice weight: {dice_weight}, Initial LR: {initial_lr}, Weight Decay: {weight_decay}, Accumulation Steps: {accumulation_steps})..."
    )
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_bce_loss, running_dice_loss = 0.0, 0.0, 0.0

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
            leave=False,
        )

        for batch_idx, batch in progress_bar:
            if batch is None:
                print("Warning: Skipping empty batch.")
                continue

            imgs, masks, filenames = batch
            imgs = imgs.to(device)
            masks = masks.to(device, dtype=torch.float) / 255.0

            # optimizer.zero_grad() # <<< Minimal change: Removed from here

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(imgs)

                if outputs.shape[2:] != masks.shape[2:]:
                    raise ValueError(f"Spatial dimension mismatch! Output: {outputs.shape}, Mask: {masks.shape}")
                if outputs.shape[1] != masks.shape[1]:
                    raise ValueError(f"Channel dimension mismatch! Output: {outputs.shape}, Mask: {masks.shape}")

                loss_bce = criterion_bce(outputs, masks.float())
                loss_dice = dice_loss(outputs, masks)
                total_loss = bce_weight * loss_bce + dice_weight * loss_dice

            if accumulation_steps > 1:
                loss_to_backward = total_loss / accumulation_steps
            else:
                loss_to_backward = total_loss

            scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero gradients after optimizer step

            if torch.isnan(total_loss) or torch.isnan(loss_bce) or torch.isnan(loss_dice):
                print(f"NaN loss encountered in epoch {epoch + 1}, batch {batch_idx}.")  # Added batch_idx
                print(f"{filenames=}")
                print(
                    f"  Loss values: total={total_loss.item():.4f}, bce={loss_bce.item():.4f}, dice={loss_dice.item():.4f}"
                )
                print("  --- Debugging Tensor Stats ---")
                print(f"  Outputs (model predictions) stats:")
                print(f"    Shape: {outputs.shape}")
                print(f"    Contains NaN: {torch.isnan(outputs).any().item()}")
                print(f"    Contains Inf: {torch.isinf(outputs).any().item()}")
                if not torch.isnan(outputs).any() and not torch.isinf(outputs).any():
                    print(
                        f"    Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}, Mean: {outputs.mean().item():.4f}, Std: {outputs.std().item():.4f}"
                    )
                print(f"  Masks (target) stats:")
                print(f"    Shape: {masks.shape}")
                print(f"    Contains NaN: {torch.isnan(masks).any().item()}")
                print(f"    Contains Inf: {torch.isinf(masks).any().item()}")
                if not torch.isnan(masks).any() and not torch.isinf(masks).any():
                    print(
                        f"    Min: {masks.min().item():.4f}, Max: {masks.max().item():.4f}, Mean: {masks.mean().item():.4f}, Std: {masks.std().item():.4f}"
                    )
                print("  -----------------------------")
                # If NaN, and it's an accumulation step, still zero grad to prevent carry-over
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)  # Use set_to_none for potential minor memory saving
                continue

            running_loss += total_loss.item()  # Log the original, un-normalized loss for this physical batch
            running_bce_loss += loss_bce.item()
            running_dice_loss += loss_dice.item()
            progress_bar.set_postfix(loss=total_loss.item(), bce=loss_bce.item(), dice=loss_dice.item())

        if len(train_dataloader) > 0 and (len(train_dataloader) % accumulation_steps != 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_bce_loss = running_bce_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_dice_loss = running_dice_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} Avg Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce_loss:.4f}, Dice: {avg_train_dice_loss:.4f}) LR: {current_lr:.6f}"
        )

        if wandb.run:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_train_loss,
                    "avg_train_bce_loss": avg_train_bce_loss,
                    "avg_train_dice_loss": avg_train_dice_loss,
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )

        scheduler.step()

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            running_val_loss = 0.0
            running_val_bce_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_dice_coeff = 0.0
            num_val_samples = 0
            if valid_dataloader and len(valid_dataloader) > 0:
                val_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]", leave=False)
                with torch.no_grad():
                    for i_val_batch, val_batch in enumerate(val_progress_bar):
                        if val_batch is None:
                            continue
                        val_imgs, val_masks, _ = val_batch
                        val_imgs = val_imgs.to(device)
                        val_masks = val_masks.to(device, dtype=torch.float) / 255.0
                        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                            val_outputs = model(val_imgs)
                            if val_outputs.shape[2:] != val_masks.shape[2:]:
                                print(
                                    f"Warning: Validation shape mismatch! Output: {val_outputs.shape}, Mask: {val_masks.shape}"
                                )
                                continue
                            if val_outputs.shape[1] != val_masks.shape[1]:
                                print(
                                    f"Warning: Validation channel mismatch! Output: {val_outputs.shape}, Mask: {val_masks.shape}"
                                )
                                continue
                            val_loss_bce = criterion_bce(val_outputs, val_masks)
                            val_loss_dice = dice_loss(val_outputs, val_masks)
                            val_total_loss = bce_weight * val_loss_bce + dice_weight * val_loss_dice
                        if not torch.isnan(val_total_loss):
                            running_val_loss += val_total_loss.item()
                            running_val_bce_loss += val_loss_bce.item()
                            running_val_dice_loss += val_loss_dice.item()
                            val_progress_bar.set_postfix(
                                loss=val_total_loss.item(), bce=val_loss_bce.item(), dice=val_loss_dice.item()
                            )
                        else:
                            print(f"Warning: NaN validation loss encountered in epoch {epoch + 1}.")
                            continue
                        with torch.no_grad():
                            val_outputs_sigmoid = torch.sigmoid(val_outputs)
                            dice_coeffs_batch = dice_coeff(val_outputs_sigmoid, val_masks, smooth=1.0, epsilon=1e-6)
                            dice_per_item = dice_coeffs_batch.mean(dim=1)
                            running_val_dice_coeff += dice_per_item.sum().item()
                            num_val_samples += val_imgs.size(0)
                        if wandb.run and wandb_log_images and i_val_batch == 0:
                            log_images = []
                            num_samples_to_log = min(wandb_num_images_to_log, val_imgs.size(0))
                            val_imgs_cpu = val_imgs[:num_samples_to_log].cpu().detach()
                            val_masks_cpu = val_masks[:num_samples_to_log].cpu().detach()
                            val_outputs_cpu = val_outputs[:num_samples_to_log].cpu().detach()
                            preds_sigmoid = torch.sigmoid(val_outputs_cpu)
                            preds_binary = (preds_sigmoid > 0.5).float()
                            for i in range(num_samples_to_log):
                                img = val_imgs_cpu[i][[2, 3, 4], ...]
                                mask_gt_combined = val_masks_cpu[i]
                                mask_pred_combined = preds_binary[i]
                                img = torch.clamp(img, 0, 1)
                                mask_gt_combined = torch.clamp(mask_gt_combined, 0, 1)
                                mask_pred_combined = torch.clamp(mask_pred_combined, 0, 1)
                                combined_image = torchvision.utils.make_grid(
                                    [img, mask_gt_combined, mask_pred_combined], nrow=3, padding=2, normalize=False
                                )
                                log_images.append(
                                    wandb.Image(
                                        combined_image, caption=f"Epoch {epoch + 1} Sample {i} (Img | GT | Pred)"
                                    )
                                )
                            if log_images:
                                wandb.log({"validation_samples": log_images}, step=epoch + 1)
                avg_val_loss = running_val_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_bce_loss = running_val_bce_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_dice_loss = running_val_dice_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else 0
                avg_val_dice = running_val_dice_coeff / num_val_samples if num_val_samples > 0 else 0.0
                print(
                    f"Epoch {epoch + 1} Avg Valid Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce_loss:.4f}, Dice: {avg_val_dice_loss:.4f}) Dice Coeff: {avg_val_dice:.4f}"
                )
                if wandb.run:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "avg_val_loss": avg_val_loss,
                            "avg_val_bce_loss": avg_val_bce_loss,
                            "avg_val_dice_loss": avg_val_dice_loss,
                            "avg_val_dice_coeff": avg_val_dice,
                        },
                        step=epoch + 1,
                    )
            else:
                print(f"Epoch {epoch + 1} Validation skipped (empty or no dataloader).")
            model.train()

        if (epoch + 1) % 10 == 0:
            print(f"--- Performing Dice Score Evaluation for Checkpoint at Epoch {epoch + 1} ---")
            model.eval()  # Set model to evaluation mode

            # Temporary variables for this specific validation pass for checkpointing
            current_epoch_dice_eval_running_coeff = 0.0
            current_epoch_dice_eval_num_samples = 0

            if valid_dataloader and len(valid_dataloader) > 0:
                dice_eval_progress_bar = tqdm(
                    valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Dice Eval for Ckpt]", leave=False
                )
                with torch.no_grad():  # Ensure no gradients are calculated
                    for val_batch_ckpt in dice_eval_progress_bar:
                        if val_batch_ckpt is None:
                            continue
                        val_imgs_ckpt, val_masks_ckpt, _ = val_batch_ckpt
                        val_imgs_ckpt = val_imgs_ckpt.to(device)
                        val_masks_ckpt = val_masks_ckpt.to(device, dtype=torch.float) / 255.0

                        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                            val_outputs_ckpt = model(val_imgs_ckpt)
                            # Optional: Add shape checks if necessary, though they should be consistent

                        # Calculate Dice Coefficient (using sigmoid output)
                        val_outputs_ckpt_sigmoid = torch.sigmoid(val_outputs_ckpt)
                        dice_coeffs_batch_ckpt = dice_coeff(
                            val_outputs_ckpt_sigmoid, val_masks_ckpt, smooth=1.0, epsilon=1e-6
                        )  # Assumes (B, C)
                        dice_per_item_ckpt = dice_coeffs_batch_ckpt.mean(
                            dim=1
                        )  # Average across classes for each sample (B,)

                        current_epoch_dice_eval_running_coeff += dice_per_item_ckpt.sum().item()
                        current_epoch_dice_eval_num_samples += val_imgs_ckpt.size(0)

                avg_dice_for_this_epoch_eval = (
                    current_epoch_dice_eval_running_coeff / current_epoch_dice_eval_num_samples
                    if current_epoch_dice_eval_num_samples > 0
                    else 0.0
                )
                print(
                    f"Epoch {epoch + 1} [Dice Eval for Ckpt]: Calculated Avg Dice Coeff = {avg_dice_for_this_epoch_eval:.4f}"
                )

                if avg_dice_for_this_epoch_eval > best_dice_score_for_checkpointing:
                    print(
                        f"New best dice score for checkpoint: {avg_dice_for_this_epoch_eval:.4f} (previous: {best_dice_score_for_checkpointing:.4f})"
                    )

                    # Remove old best checkpoint if it exists
                    if best_dice_checkpoint_path and os.path.exists(best_dice_checkpoint_path):
                        try:
                            os.remove(best_dice_checkpoint_path)
                            print(f"Removed old best dice checkpoint: {best_dice_checkpoint_path}")
                        except OSError as e:
                            print(f"Error removing old checkpoint {best_dice_checkpoint_path}: {e}")

                    best_dice_score_for_checkpointing = avg_dice_for_this_epoch_eval
                    best_dice_checkpoint_epoch = epoch + 1

                    checkpoint_filename = f"model_best_dice_epoch{best_dice_checkpoint_epoch}_dice{best_dice_score_for_checkpointing:.4f}.pth"
                    best_dice_checkpoint_path = os.path.join(output_dir, checkpoint_filename)  # Use output_dir

                    torch.save(model.state_dict(), best_dice_checkpoint_path)
                    print(f"Saved new best dice checkpoint: {best_dice_checkpoint_path}")

                    if wandb.run:
                        wandb.summary["best_dice_checkpoint_epoch"] = best_dice_checkpoint_epoch
                        wandb.summary["best_dice_checkpoint_score"] = best_dice_score_for_checkpointing
                        # Log the path as a string, not as an artifact unless specifically configured
                        wandb.summary["best_dice_checkpoint_filename"] = checkpoint_filename
            else:
                print(f"Epoch {epoch + 1} [Dice Eval for Ckpt]: Skipped (no validation dataloader or empty).")

            # Early stopping logic
            if epoch + 1 - best_dice_checkpoint_epoch >= early_stopping_threshold:
                print(
                    f"Stopping training early: No improvement in dice score for {epoch + 1 - best_dice_checkpoint_epoch} epochs."
                )
                break

            model.train()  # Ensure model is back in training mode
            print(f"--- Finished Dice Score Evaluation for Checkpoint at Epoch {epoch + 1} ---")
    print("Training finished.")
    return best_dice_checkpoint_path
