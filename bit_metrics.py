"""Bit-level metrics shared by watermark pipelines."""

import torch


def bit_error_rate(
    reference_bits: torch.Tensor,
    estimated_bits: torch.Tensor,
) -> float:
    """Compute BER between two bit tensors (fraction of mismatched bits)."""
    if reference_bits.shape != estimated_bits.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(reference_bits.shape)} vs estimated {tuple(estimated_bits.shape)}"
        )

    ref = reference_bits.detach().to(dtype=torch.int64).reshape(-1)
    est = estimated_bits.detach().to(dtype=torch.int64).reshape(-1)
    if ref.numel() == 0:
        raise ValueError("Cannot compute BER on empty bit tensors.")
    return float((ref != est).float().mean().item())


def normalized_correlation(
    reference_bits: torch.Tensor,
    estimated_bits: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute NC = <x,y> / (||x|| ||y||) between two bit tensors."""
    if reference_bits.shape != estimated_bits.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(reference_bits.shape)} vs estimated {tuple(estimated_bits.shape)}"
        )

    x = reference_bits.detach().to(dtype=torch.float32).reshape(-1)
    y = estimated_bits.detach().to(dtype=torch.float32).reshape(-1)
    if x.numel() == 0:
        raise ValueError("Cannot compute NC on empty bit tensors.")

    numerator = torch.dot(x, y)
    denominator = x.norm() * y.norm()
    if float(denominator.item()) < eps:
        return 0.0
    return float((numerator / denominator).item())
