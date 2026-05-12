import torch
import mrpro
from typing import Literal


def brain_mask(image: torch.Tensor, threshold: float = 0.1, eps: float = 1e-6):
    """
    Brain mask extraction.

    Parameters
    ----------
    image
        tensor of shape (H, W) representing a real-valued
        image (typically the magnitude for complex-valued MR images).

    threshold
        Background threshold after normalization.

    eps
        float for stabilization purposes


    Returns
    -------
    mask
        torch.Tensor of shape (H, W) representing a binary mask:
            1 -> image content (brain/head region) in our case
            0 -> background
    """
    image = (image - image.min()) / (image.max() - image.min() + eps)

    foreground = image > threshold

    ny, nx = foreground.shape

    columns = torch.arange(nx, device=image.device).view(1, nx)

    left = torch.where(foreground, columns, nx).min(dim=1).values
    right = torch.where(foreground, columns, -1).max(dim=1).values

    row_mask = (columns >= left[:, None]) & (columns <= right[:, None])

    rows = torch.arange(ny, device=image.device).view(ny, 1)

    top = torch.where(foreground, rows, ny).min(dim=0).values
    bottom = torch.where(foreground, rows, -1).max(dim=0).values

    column_mask = (rows >= top[None, :]) & (rows <= bottom[None, :])

    mask = row_mask & column_mask

    return mask.float()


def psnr(
    target: torch.Tensor,
    image: torch.Tensor,
    weight: torch.Tensor | complex = 1.0,
    reduction: Literal["full", "mean"] = "full",
) -> torch.Tensor:
    """Compute the peak signal-to-noise-ratio of an image.

    Parameters
    ----------
    target
        Ground truth 2D tensor, shape `(batch coils z y x)`.
    image
        Tensor to be compared to the target.
    weight
        A (optional) weight matrix weighting the l2-error.

    Raises:
        ValueError: if reduction is not 'full' or 'mean'

    Returns:
        torch.Tensor: the PSNR value averaged over the
            batch-dimension or a tensor containing the different PSNRs

    """

    if reduction not in ["full", "mean"]:
        raise ValueError(
            f"'reduction' must be either 'full' or 'mean', got {reduction}"
        )
    (mse_metric,) = mrpro.operators.functionals.MSE(
        target=target, weight=weight, dim=tuple(range(-(target.ndim - 1), 0))
    )(image)
    max_val = torch.amax(
        target.abs(), dim=tuple(range(target.ndim + 1, 1)), keepdim=False
    )
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse_metric))

    return psnr_value.mean() if reduction == "mean" else psnr_value
