"""PNG figures for original vs watermarked comparison."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def waveform_to_mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    """1-D numpy signal for waveform/specgram (matches batch/channel layout from models)."""
    x = waveform.squeeze().detach().cpu().numpy()
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x[0]
    raise ValueError(f"Unexpected waveform rank after squeeze: {x.ndim}")


def _plot_waveform_and_specgram_on_axes(
    ax_wave,
    ax_spec,
    waveform_1d: np.ndarray,
    sample_rate: int,
    title: str,
) -> None:
    """Same layout as facebookresearch/audioseal examples/notebook.py."""
    num_frames = waveform_1d.shape[-1]
    time_axis = np.arange(0, num_frames, dtype=np.float64) / float(sample_rate)
    ax_wave.plot(time_axis, waveform_1d, linewidth=1)
    ax_wave.set_title(f"{title} — waveform")
    ax_wave.grid(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ax_spec.specgram(waveform_1d, Fs=sample_rate)
    ax_spec.set_title(f"{title} — spectrogram")


def save_original_vs_watermarked_plot(
    wav_original: torch.Tensor,
    wav_watermarked: torch.Tensor,
    sample_rate: int,
    suptitle: str,
    out_path: str,
    *,
    dpi: int = 150,
) -> None:
    """Two rows: original (waveform + specgram), watermarked (waveform + specgram)."""
    plt.rcParams["figure.figsize"] = (20, 6)
    y0 = waveform_to_mono_numpy(wav_original)
    y1 = waveform_to_mono_numpy(wav_watermarked)
    figure, axes = plt.subplots(2, 2)
    _plot_waveform_and_specgram_on_axes(
        axes[0, 0], axes[0, 1], y0, sample_rate, title="Original audio"
    )
    _plot_waveform_and_specgram_on_axes(
        axes[1, 0], axes[1, 1], y1, sample_rate, title="Watermarked audio"
    )
    figure.suptitle(suptitle)
    figure.tight_layout()
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def close_all_figures() -> None:
    plt.close("all")
