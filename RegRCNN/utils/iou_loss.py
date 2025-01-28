import torch
from typing import Tuple
from torch import nn, Tensor
import math


def _upcast_non_float(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.dtype not in (torch.float32, torch.float64):
        return t.float()
    return t

def _loss_inter_union_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Unbind the dimensions of both sets of boxes
    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)

    # Calculate intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    zkis1 = torch.max(z1, z1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)
    zkis2 = torch.min(z2, z2g)

    # Initialize intersection volumes tensor
    intsctk = torch.zeros_like(x1)
    # Calculate mask where there is an overlap
    mask = (xkis2 > xkis1) & (ykis2 > ykis1) & (zkis2 > zkis1)
    # Calculate intersection volumes where overlap exists
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask]) * (zkis2[mask] - zkis1[mask])
    
    # Calculate union volumes
    volume1 = (x2 - x1) * (y2 - y1) * (z2 - z1)
    volume2 = (x2g - x1g) * (y2g - y1g) * (z2g - z1g)
    unionk = volume1 + volume2 - intsctk

    return intsctk, unionk

def distance_box_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Gradient-friendly IoU loss for 3D bounding boxes with an additional penalty that
    is non-zero when the distance between boxes' centers isn't zero. Indeed, for two
    exactly overlapping boxes in 3D, the distance IoU is the same as the IoU loss.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2``, ``0 <= y1 < y2``, and ``0 <= z1 < z2``. The two boxes should have the
    same dimensions.

    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[N, 6]): second set of boxes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Returns:
        Tensor: Loss tensor with the reduction option applied.
    """

    boxes1 = _upcast_non_float(boxes1)
    boxes2 = _upcast_non_float(boxes2)

    loss, _ = _diou_iou_loss_3d(boxes1, boxes2, eps)

    # Check reduction option and return loss accordingly
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")

def _diou_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:

    intsct, union = _loss_inter_union_3d(boxes1, boxes2)  # This function must handle 3D intersection and union
    iou = intsct / (union + eps)
    # smallest enclosing box
    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    zc1 = torch.min(z1, z1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    zc2 = torch.max(z2, z2g)
    # The diagonal distance of the smallest enclosing box squared
    diagonal_distance_squared = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + ((zc2 - zc1) ** 2) + eps
    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    z_p = (z2 + z1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    z_g = (z1g + z2g) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2) + ((z_p - z_g) ** 2)
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared)
    return loss, iou

def complete_box_iou_loss_3d(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Gradient-friendly IoU loss for 3D bounding boxes with an additional penalty 
    that is non-zero when the boxes do not overlap. This loss function considers 
    important geometrical factors such as overlap volume, normalized central point 
    distance, and aspect ratio of 3D boxes.
    
    Both sets of boxes are expected to be in `(x1, y1, z1, x2, y2, z2)` format with
    `0 <= x1 < x2`, `0 <= y1 < y2`, and `0 <= z1 < z2`. The two boxes should have the
    same dimensions.
    
    Args:
        boxes1 : (Tensor[N, 6] or Tensor[6]) first set of boxes
        boxes2 : (Tensor[N, 6] or Tensor[6]) second set of boxes
        reduction : (string, optional) Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`. Default: `'none'`
        eps : (float): small number to prevent division by zero. Default: 1e-7
    
    Returns:
        Tensor: Loss tensor with the reduction option applied.
    """
    diou_loss, iou = _diou_iou_loss_3d(boxes1, boxes2)

    x1, y1, z1, x2, y2, z2 = boxes1.unbind(dim=-1)
    x1g, y1g, z1g, x2g, y2g, z2g = boxes2.unbind(dim=-1)

    # Width, height, and depth of boxes
    w_pred = x2 - x1
    h_pred = y2 - y1
    d_pred = z2 - z1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    d_gt = z2g - z1g

    v = (4 / (math.pi**2)) * torch.pow((torch.atan(w_gt * h_gt / (d_gt + eps)) - torch.atan(w_pred * h_pred / (d_pred + eps))), 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    loss = diou_loss + alpha * v

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction}' \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
