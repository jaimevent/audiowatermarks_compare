"""Timestamped CSV outputs under ``<repo>/metrics/``."""

from __future__ import annotations

import csv
import os
from datetime import datetime

DETECTION_LOG_HEADER: list[str] = [
    "audio_file",
    "algorithm",
    "detected",
    "sample_rate",
    "detection_threshold",
    "file_fraction_threshold",
    "message_threshold",
    "detect_frame_fraction",
    "num_frames",
    "mean_p_wm",
    "frac_frames_above_threshold",
    "std_p_wm",
    "min_p_wm",
    "max_p_wm",
    "decoded_message_binary",
    "message_probs",
]

WATERMARK_METRICS_HEADER: list[str] = [
    "audio_file",
    "algorithm",
    "snr_db",
    "pesq_val",
    "ber_val",
    "nc_val",
]


def metrics_csv_filepath(user_path: str | None, *, default_filename: str) -> str:
    """Path under ``<repo>/metrics/`` with ``YYYYMMDD_HHMMSS`` prefixed basename."""
    if user_path and str(user_path).strip():
        base = os.path.basename(str(user_path).strip())
    else:
        base = default_filename
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(metrics_dir, f"{stamp}_{base}")


def write_watermark_metrics_header(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(WATERMARK_METRICS_HEADER)


def append_watermark_metrics_row(
    path: str,
    *,
    audio_file: str,
    algorithm: str,
    snr_db: float,
    pesq_val: float,
    ber_val: float,
    nc_val: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                audio_file,
                algorithm,
                round(snr_db, 2),
                round(pesq_val, 2),
                round(ber_val, 4),
                round(nc_val, 4),
            ]
        )


def write_attack_metrics_header(path: str, attack_metric_names: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["audio_file", "algorithm", *attack_metric_names])


def append_attack_metrics_row(
    path: str,
    *,
    audio_file: str,
    algorithm: str,
    attack_metric_names: list[str],
    attack_row: dict[str, str],
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [audio_file, algorithm, *[attack_row[name] for name in attack_metric_names]]
        )


def write_detection_log_header(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DETECTION_LOG_HEADER)
