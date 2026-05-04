"""Shared interface for watermark algorithm backends (registry-driven dispatch)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from typing import ClassVar

import torch


class WatermarkBackend(ABC):
    """One backend per algorithm id (see ``registry.ALGORITHM_REGISTRY``)."""

    name: ClassVar[str]

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @abstractmethod
    def setup(self, command: str) -> None:
        """Load models and any per-run random state (e.g. WavMark payload)."""

    @abstractmethod
    def prepare_wav_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        """Transform loaded waveform ``(channels, samples)`` before ``unsqueeze(0)``."""

    @abstractmethod
    def embed_watermark(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return ``(metrics_ref, watermarked_batched, metrics_sample_rate)``."""

    @abstractmethod
    def finalize_watermarked_cuda(self, watermarked: torch.Tensor) -> torch.Tensor:
        """Optional CUDA move after embed (AudioSeal)."""

    @abstractmethod
    def compute_ber_nc_before_save(
        self,
        wav_batched: torch.Tensor,
        watermarked: torch.Tensor,
        sample_rate: int,
    ) -> tuple[float, float] | None:
        """BER/NC before writing WAV; return ``None`` if only computed after save."""

    @abstractmethod
    def compute_ber_nc_after_save(self, wav_out_path: str) -> tuple[float, float] | None:
        """BER/NC from saved file (WavMark); return ``None`` if unused."""

    @abstractmethod
    def print_ber_line(self, ber_val: float) -> None:
        """Log BER with algorithm-appropriate wording."""

    @abstractmethod
    def run_detect(
        self,
        wav_batched: torch.Tensor,
        sample_rate: int,
        audio_file: str,
        detect_log_path: str | None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Decode / detect and append CSV row if ``detect_log_path`` is set."""

    @abstractmethod
    def attack_prepare(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        """Return ``(watermarked_batched, attack_sample_rate)`` for the attack suite."""

    @abstractmethod
    def attack_print_baseline(
        self,
        wav_wm: torch.Tensor,
        atk_sr: int,
        msg_t: float,
        det_t: float,
        frac_t: float,
    ) -> None:
        """Print baseline (no attack) robustness line."""

    @abstractmethod
    def attack_evaluate(
        self,
        wav_wm: torch.Tensor,
        attacked_wm: torch.Tensor,
        atk_sr: int,
        attack_name: str,
        msg_t: float,
        det_t: float,
        frac_t: float,
    ) -> bool:
        """Print one attack line; return whether the watermark resisted."""
