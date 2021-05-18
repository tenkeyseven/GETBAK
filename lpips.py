import lpips
import torch
import torch.nn as nn

class LPIPSLoss(nn.Module):
    def __init__(self, lpips_instance: lpips.LPIPS) -> None:
        """
        Creates a criterion that calculates a normalized LPIPS score between 2 images.
        Args:
            lpips_instance: The actual LPIPS instance to optimize against.
        """
        super(LPIPSLoss, self).__init__()
        self.lpips_instance = lpips_instance

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return normal_lpips(x, y, self.lpips_instance)


def normal_lpips(x: torch.Tensor, y: torch.Tensor, lpips_instance: lpips.LPIPS) -> torch.Tensor:
    """
    Computes the LPIPS score between the two images. Higher means further/more
        different. Lower means more similar. The input image should be normalized to
        (0, 1) because here we will normalize it to (-1, 1) for compatibility with the
        original LPIPS module.
    Args:
        x: The input image.
        y: The target image.
        lpips_instance: The actual LPIPS instance to optimize against.
    Returns:
        The LPIPS score between the two images.
    """
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError(f"Expected 2 torch tensors but got {type(x)} and {type(y)}")
    if x.device != y.device:
        raise TypeError(f"Expected 2 tensors on the same device but got {x.device} and {y.device}")
    if x.shape != y.shape:
        raise TypeError(f"Expected 2 tensors of equal shape but got {x.shape} and {y.shape}.")

    # TODO: add normalization process to convert image from (0, 1) to (-1, 1)
    normalized_x, normalized_y = (x - 0.5) / 0.5, (y - 0.5) / 0.5  # nope, not working

    return lpips_instance.forward(normalized_x, normalized_y)