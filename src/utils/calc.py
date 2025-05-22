import torch

def dice_coeff(pred, target, smooth=1.0, epsilon=1e-6):
    """Calculates Dice Coefficient per class."""
    # pred: (N, C, H, W), target: (N, C, H, W)
    # Apply sigmoid to predictions
    pred = torch.sigmoid(pred)

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # (N, C, H*W)
    target_flat = target.view(target.shape[0], target.shape[1], -1)  # (N, C, H*W)

    intersection = (pred_flat * target_flat).sum(2)  # (N, C)
    pred_sum = pred_flat.sum(2)  # (N, C)
    target_sum = target_flat.sum(2)  # (N, C)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth + epsilon)  # (N, C)

    return dice  # Return per-class dice score for the batch


def dice_loss(pred, target, smooth=1.0, epsilon=1e-6):
    """Calculates Dice Loss (average over classes)."""
    dice_coeffs = dice_coeff(pred, target, smooth, epsilon)  # (N, C)
    # Average dice score across classes, then subtract from 1
    return 1.0 - dice_coeffs.mean()
