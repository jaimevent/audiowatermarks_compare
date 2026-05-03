"""PNG figures for original vs watermarked comparison and attack robustness summaries."""

from __future__ import annotations

import os
import warnings
from typing import Sequence

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


def _short_label(text: str, max_len: int = 42) -> str:
    """Keep axis labels readable in dense heatmaps/bar charts."""
    t = text.replace("\\", "/")
    return t if len(t) <= max_len else "…" + t[-(max_len - 1) :]


def attack_row_values_to_binary(attack_row: dict[str, str], attack_metric_names: Sequence[str]) -> list[int]:
    """Map resisted ``'X'`` / failed ``'-'`` CSV cells to 1 / 0 for plotting."""
    return [1 if str(attack_row.get(name, "-")).strip().upper() == "X" else 0 for name in attack_metric_names]


def save_attack_summary_heatmap(
    resist_matrix: np.ndarray | Sequence[Sequence[float | int]],
    audio_labels: Sequence[str],
    attack_names: Sequence[str],
    *,
    algorithm: str,
    out_path: str,
    dpi: int = 150,
    title_suffix: str = "",
) -> None:
    """Rows = audio clips, columns = attacks; cells = 1 if watermark resisted else 0."""
    matrix = np.asarray(resist_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"resist_matrix must be 2-D; got shape {matrix.shape}")
    if matrix.shape[1] != len(attack_names):
        raise ValueError(
            f"Matrix columns {matrix.shape[1]} != len(attack_names) {len(attack_names)}"
        )
    if matrix.shape[0] != len(audio_labels):
        raise ValueError(
            f"Matrix rows {matrix.shape[0]} != len(audio_labels) {len(audio_labels)}"
        )

    n_rows, n_cols = matrix.shape
    fig_w = min(26.0, max(10.0, 2.8 + n_cols * 0.52))
    fig_h = min(36.0, max(6.0, 3.8 + n_rows * 0.38))

    plt.rcParams["figure.figsize"] = (fig_w, fig_h)
    figure, axis = plt.subplots()
    mesh = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn", interpolation="nearest")
    axis.set_xticks(np.arange(n_cols))
    axis.set_yticks(np.arange(n_rows))
    axis.set_xticklabels([_short_label(n, 28) for n in attack_names], rotation=75, ha="right")
    axis.set_yticklabels([_short_label(l, 44) for l in audio_labels])
    axis.set_xlabel("Attack")
    axis.set_ylabel("Audio clip")
    axis.set_title(
        f"{algorithm}: attack robustness — cell = 1 if watermark resisted{title_suffix}".strip()
    )
    figure.colorbar(mesh, ax=axis, fraction=0.025, pad=0.02, label="Resisted (1) / failed (0)")
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def save_attack_summary_bar_chart(
    resist_matrix: np.ndarray | Sequence[Sequence[float | int]],
    attack_names: Sequence[str],
    *,
    algorithm: str,
    out_path: str,
    dpi: int = 150,
    title_suffix: str = "",
) -> None:
    """Vertical bars: mean resistance rate per attack (over clips in ``resist_matrix``)."""
    matrix = np.asarray(resist_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"resist_matrix must be 2-D; got shape {matrix.shape}")
    if matrix.shape[1] != len(attack_names):
        raise ValueError(
            f"Matrix columns {matrix.shape[1]} != len(attack_names) {len(attack_names)}"
        )

    n_clips = max(matrix.shape[0], 1)
    means = np.mean(matrix, axis=0)

    plt.rcParams["figure.figsize"] = (min(26.0, max(10.0, len(attack_names) * 0.7)), 6.8)
    figure, axis = plt.subplots()
    xs = np.arange(len(attack_names))
    bars = axis.bar(xs, means, color="steelblue", edgecolor="gray", linewidth=0.6)
    axis.axhline(y=0.5, linestyle="--", color="orange", linewidth=1.0, label="50% resisted")
    axis.set_xticks(xs)
    axis.set_xticklabels([_short_label(n, 32) for n in attack_names], rotation=72, ha="right")
    axis.set_ylim(0.0, 1.09)
    axis.set_ylabel("Fraction of clips that resisted attack")
    axis.set_xlabel("Attack")
    n_attacks = len(attack_names)
    axis.set_title(
        f"{algorithm}: mean robustness vs attack (n clips = {n_clips}; n attacks = {n_attacks}){title_suffix}".strip()
    )
    axis.legend(loc="upper right", fontsize="small")

    # Label bar tops when few attacks
    if n_attacks <= 18:
        for bar, rate in zip(bars, means, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(rate + 0.03, 1.02),
                f"{rate:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
