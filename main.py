#!/usr/bin/python

import os

# AudioSeal wraps the encoder with torch.compile; Inductor on Windows needs MSVC (cl.exe).
# Without Build Tools, compilation fails. Disable Dynamo before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import csv
import re
from datetime import datetime
from typing import Callable
import shutil
import subprocess
import sys
import tempfile
import warnings
import torch

torch._dynamo.config.disable = True  # belt-and-suspenders if env above is ignored

import numpy as np
import soundfile as sf
from audioseal import AudioSeal

import wavmark as wm
from wavmark.utils import wm_add_util as wavmark_wm_add_util
from wavmark_io import (
    wavmark_decode_sliding_stats,
    wavmark_decode_watermark,
    wavmark_mono_16k_tensor,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pesq import pesq
import librosa

from attacks import AudioEffects


def metrics_csv_filepath(user_path: str | None, *, default_filename: str) -> str:
    """Path under `<repo>/metrics/` with ``YYYYMMDD_HHMMSS`` prefixed basename."""
    if user_path and str(user_path).strip():
        base = os.path.basename(str(user_path).strip())
    else:
        base = default_filename
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(metrics_dir, f"{stamp}_{base}")


def convert_flac_to_wav(src_flac: str, dst_wav: str) -> None:
    """Decode FLAC and write a WAV container (float32 samples). libsndfile handles FLAC."""
    data, samplerate = sf.read(src_flac, dtype="float32", always_2d=True)
    sf.write(dst_wav, data, samplerate, format="WAV", subtype="FLOAT")


def _load_from_wav_file(wav_path: str) -> tuple[torch.Tensor, int]:
    # soundfile avoids torchaudio 2.10+ routing through torchcodec for simple file decode.
    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy())
    return wav, sample_rate


def load_audio(file_path: str) -> tuple[torch.Tensor, int]:
    """Load audio for the model. WAV is read directly; FLAC is converted to WAV then loaded."""
    if file_path.lower().endswith(".flac"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            convert_flac_to_wav(file_path, tmp_path)
            return _load_from_wav_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    return _load_from_wav_file(file_path)

def _to_mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    """Match notebook expectations: 1D signal for waveform and specgram."""
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
    # Matplotlib specgram uses log10(power); silent bins → divide-by-zero warnings.
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
    y0 = _to_mono_numpy(wav_original)
    y1 = _to_mono_numpy(wav_watermarked)
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


def save_watermarked_wav(
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    out_path: str,
) -> None:
    """Write watermarked audio as WAV (float32). Expects shape (batch, channels, samples)."""
    if watermarked_audio.dim() != 3:
        raise ValueError(
            f"Expected watermarked tensor rank 3 (batch, channels, samples); got {watermarked_audio.dim()}"
        )
    # (channels, samples) -> (samples, channels) for soundfile
    data = watermarked_audio[0].detach().cpu().numpy().T.astype(np.float32, copy=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, data, sample_rate, format="WAV", subtype="FLOAT")


def watermarking_snr_db(
    wav_original: torch.Tensor,
    wav_watermarked: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> float:
    """SNR (dB) of the original versus the additive watermark.

    Uses the usual embedding SNR: signal = original, noise = watermarked − original.
    Averages mean square across channels and time (batch must match; only index 0 is used).
    """
    if wav_original.shape != wav_watermarked.shape:
        raise ValueError(
            f"Shape mismatch: original {tuple(wav_original.shape)} vs watermarked {tuple(wav_watermarked.shape)}"
        )
    if wav_original.dim() != 3:
        raise ValueError(
            f"Expected tensors of rank 3 (batch, channels, samples); got {wav_original.dim()}"
        )
    original = wav_original[0]
    noise = wav_watermarked[0] - original
    power_signal = (original * original).mean().clamp_min(eps)
    power_noise = (noise * noise).mean().clamp_min(eps)
    return (10.0 * torch.log10(power_signal / power_noise)).item()

def pesq_score(
    wav_original: torch.Tensor,
    wav_watermarked: torch.Tensor,
    sample_rate: int = 16000,
) -> float:
    """PESQ score of the original versus the watermarked audio."""
    if wav_original.shape != wav_watermarked.shape:
        raise ValueError(
            f"Shape mismatch: original {tuple(wav_original.shape)} vs watermarked {tuple(wav_watermarked.shape)}"
        )
    if wav_original.dim() != 3:
        raise ValueError(
            f"Expected tensors of rank 3 (batch, channels, samples); got {wav_original.dim()}"
        )
    # PESQ expects 1-D numpy arrays (mono float32).
    ref = _to_mono_numpy(wav_original[0]).astype(np.float32, copy=False)
    deg = _to_mono_numpy(wav_watermarked[0]).astype(np.float32, copy=False)

    # PESQ only supports 8 kHz (nb) and 16 kHz (wb): resample other rates to 16 kHz.
    if sample_rate not in (8000, 16000):
        ref = _resample_audio(ref, sample_rate, 16000)
        deg = _resample_audio(deg, sample_rate, 16000)
        sample_rate = 16000

    mode = "wb" if sample_rate == 16000 else "nb"
    return float(pesq(sample_rate, ref, deg, mode))

# For the moment not used due to the lack of a PEAQ backend in PATH
def odg_score(
    wav_original: torch.Tensor,
    wav_watermarked: torch.Tensor,
    sample_rate: int,
) -> float:
    """ODG score via an external PEAQ backend (gstpeaq/peaqb).

    ODG is defined in ITU-R BS.1387 (PEAQ) and typically ranges in [-4, 0].
    """
    if wav_original.shape != wav_watermarked.shape:
        raise ValueError(
            f"Shape mismatch: original {tuple(wav_original.shape)} vs watermarked {tuple(wav_watermarked.shape)}"
        )
    if wav_original.dim() != 3:
        raise ValueError(
            f"Expected tensors of rank 3 (batch, channels, samples); got {wav_original.dim()}"
        )

    ref = _to_mono_numpy(wav_original[0]).astype(np.float32, copy=False)
    deg = _to_mono_numpy(wav_watermarked[0]).astype(np.float32, copy=False)

    # Most open PEAQ tools expect 48 kHz input.
    if sample_rate != 48000:
        ref = _resample_audio(ref, sample_rate, 48000)
        deg = _resample_audio(deg, sample_rate, 48000)
        sample_rate = 48000

    backend = shutil.which("gstpeaq") or shutil.which("peaqb")
    if backend is None:
        raise RuntimeError(
            "No PEAQ backend found. Install 'gstpeaq' or 'peaqb' and make it available in PATH."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "ref.wav")
        deg_path = os.path.join(tmpdir, "deg.wav")
        # PCM_16 maximizes compatibility with command-line PEAQ tools.
        sf.write(ref_path, ref, sample_rate, format="WAV", subtype="PCM_16")
        sf.write(deg_path, deg, sample_rate, format="WAV", subtype="PCM_16")

        candidate_cmds = []
        exe_name = os.path.basename(backend).lower()
        if "gstpeaq" in exe_name:
            candidate_cmds = [
                [backend, "--basic", ref_path, deg_path],
                [backend, ref_path, deg_path],
            ]
        else:
            candidate_cmds = [[backend, ref_path, deg_path]]

        output = ""
        for cmd in candidate_cmds:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode == 0:
                break
        else:
            raise RuntimeError(
                f"PEAQ backend execution failed. Last output:\n{output.strip()}"
            )

    match = re.search(
        r"(?:ODG|Objective Difference Grade)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
        output,
        flags=re.IGNORECASE,
    )
    if not match:
        raise RuntimeError(
            f"Could not parse ODG from backend output:\n{output.strip()}"
        )
    return float(match.group(1))

def bit_error_rate(
    reference_bits: torch.Tensor,
    estimated_bits: torch.Tensor,
) -> float:
    """Compute BER between two bit tensors (fraction of mismatched bits)."""
    if reference_bits.shape != estimated_bits.shape:
        raise ValueError(
            f"Shape mismatch: reference {tuple(reference_bits.shape)} vs estimated {tuple(estimated_bits.shape)}"
        )

    # print(f"[DEBUG] Reference bits: {reference_bits[:20]}")
    # print(f"[DEBUG] Estimated bits: {estimated_bits[:20]}")

    ref = reference_bits.detach().to(dtype=torch.int64).reshape(-1)
    est = estimated_bits.detach().to(dtype=torch.int64).reshape(-1)
    if ref.numel() == 0:
        raise ValueError("Cannot compute BER on empty bit tensors.")
    return float((ref != est).float().mean().item())

def bit_accuracy(
    reference_bits: torch.Tensor,
    estimated_bits: torch.Tensor,
) -> float:
    """Compute bit accuracy between two bit tensors (fraction of matched bits)."""
    return 1 - bit_error_rate(reference_bits, estimated_bits)

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

def _resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D numpy audio signal to the target sample rate."""
    if original_sr == target_sr:
        return audio

    resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return np.asarray(resampled, dtype=np.float32)


def _numpy_1d_to_batch_tensor(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).unsqueeze(0)


def print_watermark_detection_summary(
    *,
    detect_frame_fraction: torch.Tensor,
    binary_message: torch.Tensor,
    frame_logits: torch.Tensor,
    message_probs: torch.Tensor,
    detection_threshold: float,
    file_fraction_threshold: float,
    show_raw: bool,
) -> None:
    """Explain AudioSeal outputs in plain language.

    ``detect_watermark`` (upstream) sets ``detect_frame_fraction`` to the fraction of
    time frames where P(watermark) > ``detection_threshold``. The file-level verdict
    below treats the clip as detected if that fraction is >= ``file_fraction_threshold``.

    ``frame_logits`` is ``batch x 2 x frames`` with softmax over the first two channels
    (not watermarked vs watermarked). ``message_probs`` is ``batch x nbits`` in [0, 1].
    """
    frac = float(detect_frame_fraction.reshape(-1)[0].item())
    file_detected = frac >= file_fraction_threshold
    verdict = "watermark likely present" if file_detected else "watermark likely NOT present"
    print(
        f"  Detector file verdict: {verdict} "
        f"({frac:.1%} of frames with P(wm) > {detection_threshold}; "
        f"file threshold {file_fraction_threshold:.0%})"
    )

    wm_probs = frame_logits[:, 1, :].detach().float()
    mean_p = float(wm_probs.mean().item())
    frac_frames = float((wm_probs > detection_threshold).float().mean().item())
    print(
        f"  Detector frame stats: mean P(wm)={mean_p:.3f}; "
        f"{frac_frames:.1%} of frames > {detection_threshold}"
    )

    bits = binary_message[0].detach().cpu().numpy().tolist()
    if bits:
        bit_str = "".join(str(int(b)) for b in bits)
        print(f"  Decoded message (binary, threshold {detection_threshold}): {bit_str}")

    if show_raw:
        print(f"    [debug] detect_frame_fraction tensor: {detect_frame_fraction}")
        print(f"    [debug] frame P(wm) sample (first 12): {wm_probs[0, :12].cpu().tolist()}")
        print(f"    [debug] message bit probabilities: {message_probs[0].detach().cpu().numpy()}")


def watermark_file_detected(
    detector: torch.nn.Module,
    wav_watermarked: torch.Tensor,
    *,
    message_threshold: float = 0.5,
    detection_threshold: float = 0.5,
    file_fraction_threshold: float = 0.5,
) -> tuple[bool, float]:
    detect_frame_fraction, _ = detector.detect_watermark(
        wav_watermarked,
        message_threshold=message_threshold,
        detection_threshold=detection_threshold,
    )
    frac = float(detect_frame_fraction.reshape(-1)[0].item())
    return frac >= file_fraction_threshold, frac


def wavmark_payload_survives_attack(
    wmmodel,
    payload_embedded: np.ndarray,
    wav_batch_ch_first: torch.Tensor,
    audio_sample_rate: int,
) -> bool:
    """True iff WavMark decodes ``payload_embedded`` exactly from ``wav_batch_ch_first`` (rate ``audio_sample_rate``)."""

    mono16 = wavmark_mono_16k_tensor(wav_batch_ch_first, audio_sample_rate)
    dec, _, _ = wavmark_decode_watermark(wmmodel, mono16, show_progress=False)
    if dec is None:
        return False
    a = np.asarray(payload_embedded).reshape(-1).astype(np.int64)
    b = np.asarray(dec).reshape(-1).astype(np.int64)
    return bool(np.array_equal(a, b))


def print_wavmark_detection_summary(
    *,
    payload_decoded: np.ndarray | None,
    decode_info: dict,
    wav_16k: np.ndarray,
    show_raw: bool,
) -> None:
    total_w, sync_frac, sims = wavmark_decode_sliding_stats(wav_16k, decode_info)
    file_ok = payload_decoded is not None
    verdict = "watermark likely present" if file_ok else "watermark likely NOT present"
    print(
        f"  WavMark file verdict: {verdict} "
        "(payload extracted by the library decoder; waveform can still carry a watermark when BER ≠ 100%)"
    )
    print(
        f"  Sliding-window diagnostic — sync hits / windows = {sync_frac:.4f} "
        f"({len(sims)} hits over {total_w} offsets; low values are normal on long clips)"
    )
    print(f"  WavMark sliding windows: {total_w}; windows with valid sync pattern: {len(sims)}")
    if sims:
        sm = float(np.mean(sims))
        print(
            f"  WavMark sync similarity: mean={sm:.3f}; "
            f"min={min(sims):.3f}; max={max(sims):.3f}"
        )
    if payload_decoded is not None:
        bits = np.asarray(payload_decoded).reshape(-1).astype(int).tolist()
        bit_str = "".join(str(b) for b in bits)
        print(f"  Decoded payload (16 bits, hard decision): {bit_str}")
    else:
        print("  Decoded payload: (none — no reliable extraction)")
    if show_raw:
        print(f"    [debug] decode_info keys: {list(decode_info.keys())}")
        if decode_info.get("results"):
            print(f"    [debug] first hit start_sample: {decode_info['results'][0].get('start_position')}")


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


def detection_log_csv_row_audioseal(
    audio_file: str,
    algorithm: str,
    *,
    sample_rate: int,
    message_threshold: float,
    detection_threshold: float,
    file_fraction_threshold: float,
    detect_frame_fraction: torch.Tensor,
    binary_message: torch.Tensor,
    frame_logits: torch.Tensor,
    message_probs: torch.Tensor,
) -> list:
    """One row for :data:`DETECTION_LOG_HEADER`; ``detected`` is ``X`` or ``-``."""
    frac = float(detect_frame_fraction.reshape(-1)[0].item())
    detected = "X" if frac >= file_fraction_threshold else "-"
    wm_probs = frame_logits[:, 1, :].detach().float()
    n_frames = int(wm_probs.shape[1])
    mean_p = float(wm_probs.mean().item()) if n_frames > 0 else float("nan")
    frac_frames = (
        float((wm_probs > detection_threshold).float().mean().item())
        if n_frames > 0
        else float("nan")
    )
    std_p = float(wm_probs.std(unbiased=False).item()) if n_frames > 1 else 0.0
    min_p = float(wm_probs.min().item()) if n_frames > 0 else float("nan")
    max_p = float(wm_probs.max().item()) if n_frames > 0 else float("nan")
    bits = binary_message[0].detach().cpu().numpy().tolist() if binary_message.numel() else []
    bit_str = "".join(str(int(b)) for b in bits) if bits else ""
    probs_np = message_probs[0].detach().cpu().float().numpy()
    probs_str = ";".join(f"{float(x):.6f}" for x in probs_np.tolist())
    return [
        audio_file,
        algorithm,
        detected,
        sample_rate,
        detection_threshold,
        file_fraction_threshold,
        message_threshold,
        round(frac, 6),
        n_frames,
        round(mean_p, 6),
        round(frac_frames, 6),
        round(std_p, 6),
        round(min_p, 6),
        round(max_p, 6),
        bit_str,
        probs_str,
    ]


def detection_log_csv_row_wavmark(
    audio_file: str,
    algorithm: str,
    *,
    sample_rate: int,
    message_threshold: float,
    detection_threshold: float,
    file_fraction_threshold: float,
    wav_16k: np.ndarray,
    payload_decoded: np.ndarray | None,
    decode_info: dict,
) -> list:
    """CSV row aligned with :data:`DETECTION_LOG_HEADER` using WavMark decode statistics.

    ``detected`` is ``X`` when the library extracts a payload (``payload_decoded is not None``).
    Sliding-window stats are informational; unlike AudioSeal they are not calibrated to
    ``--file-fraction-threshold`` values (typically ≪0.5 on long clips even when decoding works).
    """

    total_w, sync_frac, sims = wavmark_decode_sliding_stats(wav_16k, decode_info)
    detected = "X" if payload_decoded is not None else "-"
    n_frames = total_w
    if sims:
        mean_sync = float(np.mean(sims))
        std_sync = float(np.std(sims, ddof=0)) if len(sims) > 1 else 0.0
        min_sync = float(np.min(sims))
        max_sync = float(np.max(sims))
    else:
        mean_sync = float("nan")
        std_sync = 0.0
        min_sync = float("nan")
        max_sync = float("nan")

    frac_frames = sync_frac
    detect_frame_frac = sync_frac

    if payload_decoded is not None:
        pat = np.asarray(wavmark_wm_add_util.fix_pattern[:16], dtype=int)
        payload_bits = np.asarray(payload_decoded).reshape(-1).astype(int)
        full = np.concatenate([pat, payload_bits])
        bit_str = "".join(str(int(b)) for b in full.tolist())
    else:
        bit_str = ""

    return [
        audio_file,
        algorithm,
        detected,
        sample_rate,
        detection_threshold,
        file_fraction_threshold,
        message_threshold,
        round(detect_frame_frac, 6),
        n_frames,
        round(mean_sync, 6),
        round(frac_frames, 6),
        round(std_sync, 6),
        round(min_sync, 6),
        round(max_sync, 6),
        bit_str,
        "",
    ]


def attack_eval_specs(sample_rate: int) -> list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    """(name, fn) pairs: ``fn(watermarked_batch) -> attacked_batch``."""
    fx = AudioEffects
    sr = sample_rate
    return [
        ("identity", lambda t: fx.identity(t)),
        ("updownresample", lambda t: fx.updownresample(t, sample_rate=sr)),
        (
            "random_noise",
            lambda t: fx.random_noise(t, noise_std=0.001),
        ),
        ("pink_noise", lambda t: fx.pink_noise(t, noise_std=0.01)),
        ("echo", lambda t: fx.echo(t, sample_rate=sr)),
        (
            "lowpass_5000_hz",
            lambda t: fx.lowpass_filter(t, cutoff_freq=5000, sample_rate=sr),
        ),
        (
            "highpass_500_hz",
            lambda t: fx.highpass_filter(t, cutoff_freq=500, sample_rate=sr),
        ),
        (
            "bandpass_300_8000_hz",
            lambda t: fx.bandpass_filter(
                t,
                cutoff_freq_low=300,
                cutoff_freq_high=8000,
                sample_rate=sr,
            ),
        ),
        ("smooth", lambda t: fx.smooth(t)),
        ("boost_audio_20pct", lambda t: fx.boost_audio(t, amount=20)),
        ("duck_audio_20pct", lambda t: fx.duck_audio(t, amount=20)),
        ("shush", lambda t: fx.shush(t)),
        # Output length may differ from input; detector still consumes variable-length tensors.
        ("speed_random", lambda t: fx.speed(t, speed_range=(0.5, 1.5), sample_rate=sr)),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AudioSeal utilities. Subcommands: watermark (default pipeline), detect, attack, plot."
        )
    )
    parser.add_argument(
        "--generator",
        default="audioseal_wm_16bits",
        metavar="NAME",
        help="AudioSeal generator card name (default: audioseal_wm_16bits).",
    )
    parser.add_argument(
        "--detector",
        default="audioseal_detector_16bits",
        metavar="NAME",
        help="AudioSeal detector card name (default: audioseal_detector_16bits).",
    )
    parser.add_argument(
        "--debug",
        default=True,
        help="Show debug information.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
        help="Action to run.",
    )

    input_help = "Directory containing .wav and/or .flac files (non-recursive)."

    def attach_algorithm_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--algorithm",
            "-a",
            default="audioseal",
            choices=("audioseal", "wavmark"),
            metavar="ALGORITHM",
            help="Watermark algorithm (default: %(default)s).",
        )
        p.add_argument(
            "--all-algorithms",
            action="store_true",
            dest="run_all_algorithms",
            help="Process both audioseal and wavmark.",
        )

    p_watermark = subparsers.add_parser(
        "watermark",
        help="Embed a watermark into audio files.",
    )
    p_watermark.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="DIR",
        help=input_help,
    )
    attach_algorithm_arguments(p_watermark)
    p_watermark.add_argument(
        "--output-plot",
        "-o",
        default=None,
        metavar="DIR",
        help="Directory for PNG plots (default: <input>/audioseal_plots).",
    )
    p_watermark.add_argument(
        "--output-metrics",
        "-m",
        default=None,
        metavar="FILE",
        help=(
            "Base name for the metrics CSV in the project `metrics/` folder "
            "(default: watermark_metrics.csv; final name is prefixed with timestamp)."
        ),
    )
    p_watermark.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing comparison plots.",
    )
    p_watermark.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        metavar="N",
        help="Resolution of saved PNG figures (default: 150).",
    )
    p_watermark.add_argument(
        "--output-watermarked",
        default=None,
        metavar="DIR",
        help="Directory for watermarked WAV files (default: <input>/watermarked_wav).",
    )

    p_detect = subparsers.add_parser(
        "detect",
        help="Detect a watermark in audio files.",
    )
    p_detect.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="DIR",
        help=input_help,
    )
    p_detect.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="Frame-level P(watermark) threshold (default: 0.5).",
    )
    p_detect.add_argument(
        "--message-threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="Message bit threshold passed to detect_watermark (default: 0.5).",
    )
    p_detect.add_argument(
        "--file-fraction-threshold",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="Watermark counts as detected if fraction of frames above "
        "--detection-threshold is >= this value (default: 0.5).",
    )
    p_detect.add_argument(
        "--output-detect-log",
        default=None,
        metavar="FILE",
        help=(
            "Base name for detection log CSV in the project `metrics/` folder "
            "(default: detection_log.csv; final name is prefixed with timestamp)."
        ),
    )
    attach_algorithm_arguments(p_detect)

    p_attack = subparsers.add_parser(
        "attack",
        help="Embed watermark, apply attacks from attacks.AudioEffects, then run the detector.",
    )
    p_attack.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="DIR",
        help=input_help,
    )
    p_attack.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="Frame-level P(watermark) threshold (default: 0.5).",
    )
    p_attack.add_argument(
        "--message-threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="Message bit threshold passed to detect_watermark (default: 0.5).",
    )
    p_attack.add_argument(
        "--file-fraction-threshold",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="Watermark counts as detected if fraction of frames above "
        "--detection-threshold is >= this value (default: 0.5).",
    )
    p_attack.add_argument(
        "--attack-seed",
        type=int,
        default=42,
        metavar="N",
        help="RNG seed before each stochastic attack (echo, smooth, speed, …) (default: 42).",
    )
    p_attack.add_argument(
        "--save-attacked",
        default=None,
        metavar="DIR",
        help="Optional directory to write attacked watermarked WAVs (one subfolder per file).",
    )
    p_attack.add_argument(
        "--output-attack-metrics",
        default=None,
        metavar="FILE",
        help=(
            "Base name for attack metrics CSV in the project `metrics/` folder "
            "(default: attack_metrics.csv; final name is prefixed with timestamp)."
        ),
    )
    attach_algorithm_arguments(p_attack)

    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    audio_folder = os.path.abspath(args.input)
    if not os.path.isdir(audio_folder):
        raise SystemExit(f"Not a directory: {audio_folder}")

    if getattr(args, "run_all_algorithms", False):
        algorithms = ["audioseal", "wavmark"]
    else:
        algorithms = [args.algorithm]

    print(f" -- Using the following algorithms: {algorithms} --")
    
    plot_dir = None
    watermark_metrics_csv_path = None

    if args.command == "watermark" and not args.no_plots:
        plot_dir = os.path.abspath(
            args.output_plot or os.path.join("plots")
        )
        os.makedirs(plot_dir, exist_ok=True)

    if args.command == "watermark":
        for algorithm in algorithms:
            watermarked_wav_dir = os.path.abspath(
                args.output_watermarked or os.path.join(audio_folder, f"{algorithm}_watermarked_wav")
            )
            os.makedirs(watermarked_wav_dir, exist_ok=True)

            watermark_metrics_csv_path = metrics_csv_filepath(
                args.output_metrics,
                default_filename=f"{algorithm}_watermark_metrics.csv",
            )
            with open(watermark_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["audio_file", "algorithm", "snr_db", "pesq_val", "ber_val", "nc_val"])
            print(f"{algorithm} watermark metrics CSV: {watermark_metrics_csv_path}")

    attack_metric_names: list[str] = []
    attack_metrics_csv_path = None
    if args.command == "attack":
        for algorithm in algorithms:
            attack_metric_names = [name for name, _ in attack_eval_specs(sample_rate=16000)]
            attack_metrics_csv_path = metrics_csv_filepath(
                args.output_attack_metrics,
                default_filename=f"{algorithm}_attack_metrics.csv",
            )
            with open(attack_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["audio_file", "algorithm", *attack_metric_names])
            print(f"{algorithm} attack metrics CSV: {attack_metrics_csv_path}")

    detect_log_by_algorithm: dict[str, str] = {}
    if args.command == "detect":
        for algorithm in algorithms:
            path = metrics_csv_filepath(
                args.output_detect_log,
                default_filename=f"{algorithm}_detection_log.csv",
            )
            detect_log_by_algorithm[algorithm] = path
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(DETECTION_LOG_HEADER)
            print(f"{algorithm} detection log CSV: {path}")

    audio_files = [
        f
        for f in os.listdir(audio_folder)
        if f.lower().endswith(".wav") or f.lower().endswith(".flac")
    ]

    for algorithm in algorithms:
        if algorithm == "audioseal":
            asmodel = AudioSeal.load_generator(args.generator)
            asmodel.eval()
            detector = AudioSeal.load_detector(args.detector)
            detector.eval()

            # Move models to GPU if available (tensors must match detector device).
            if torch.cuda.is_available():
                asmodel = asmodel.cuda()
                detector = detector.cuda()
        elif algorithm == "wavmark":
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            wmmodel = wm.load_model().to(device)

            if args.command in ("watermark", "attack"):
                payload = np.random.choice([0, 1], size=16)
                print("Payload:", payload)

        print(f"You've selected the following command: {args.command}")

        for audio_file in audio_files:
            print(f"---- Processing audio file: {audio_file} for {algorithm} ----")
            audio_path = os.path.join(audio_folder, audio_file)
            wav, sample_rate = load_audio(audio_path)
            # AudioSeal expects tensors on GPU; WavMark encode/decode use 1-D numpy at 16 kHz on CPU.
            if algorithm == "audioseal" and torch.cuda.is_available():
                wav = wav.cuda()

            # load_audio returns wav shaped (channels, samples) (from soundfile: channels first). After wav.unsqueeze(0) it returns 
            # (1, channels, samples): batch size 1, which matches what AudioSeal expects (batch × channels × samples).
            wav = wav.unsqueeze(0)

            # Add watermark
            if args.command == "watermark":
                if algorithm == "audioseal":
                    watermark = asmodel.get_watermark(wav)
                    watermarked_audio = wav + watermark
                    metrics_ref = wav
                    metrics_sr = sample_rate
                elif algorithm == "wavmark":
                    wav_16k = wavmark_mono_16k_tensor(wav, sample_rate)
                    watermarked_np, _ = wm.encode_watermark(
                        wmmodel, wav_16k, payload, show_progress=False
                    )
                    watermarked_audio = _numpy_1d_to_batch_tensor(watermarked_np)
                    metrics_ref = _numpy_1d_to_batch_tensor(wav_16k)
                    metrics_sr = 16000
                    if torch.cuda.is_available():
                        watermarked_audio = watermarked_audio.cuda()
                        metrics_ref = metrics_ref.cuda()

                if algorithm == "audioseal" and torch.cuda.is_available():
                    watermarked_audio = watermarked_audio.cuda()
                
                # ------- Start of metrics -------
                # Measure SNR
                snr_db = watermarking_snr_db(metrics_ref, watermarked_audio)
                print(f"  SNR: {snr_db:.2f} dB")

                # Measure PESQ
                pesq_val = pesq_score(metrics_ref, watermarked_audio, metrics_sr)
                print(f"  PESQ: {pesq_val:.2f}")

                # BER / NC: AudioSeal from detector; WavMark only after writing + reading the WAV
                # (same path as ``detect``) so metrics match what you get from the saved file.
                ber_val = 0.0
                nc_val = 0.0
                if algorithm == "audioseal":
                    _, bits_original = detector.detect_watermark(
                        wav,
                        message_threshold=0.5,
                        detection_threshold=0.5,
                    )
                    _, bits_watermarked = detector.detect_watermark(
                        watermarked_audio,
                        message_threshold=0.5,
                        detection_threshold=0.5,
                    )
                    ber_val = bit_error_rate(bits_original, bits_watermarked)
                    nc_val = normalized_correlation(bits_original, bits_watermarked)

                stem, _ = os.path.splitext(audio_file)
                wav_out_path = os.path.join(
                    watermarked_wav_dir, f"{stem}_{algorithm}_watermarked.wav"
                )
                save_watermarked_wav(watermarked_audio, metrics_sr, wav_out_path)
                print(f"  Saved watermarked WAV: {stem}_{algorithm}_watermarked.wav")

                if algorithm == "wavmark":
                    wav_disk, sr_disk = load_audio(wav_out_path)
                    payload_decoded_file, _, _ = wavmark_decode_watermark(
                        wmmodel,
                        wavmark_mono_16k_tensor(wav_disk.unsqueeze(0), sr_disk),
                        show_progress=False,
                    )
                    if payload_decoded_file is None:
                        ber_val = 100.0
                        nc_val = 0.0
                    else:
                        ber_val = float(
                            (payload != payload_decoded_file).mean() * 100.0
                        )
                        nc_val = normalized_correlation(
                            torch.from_numpy(
                                np.asarray(payload, dtype=np.float32).reshape(-1)
                            ),
                            torch.from_numpy(
                                np.asarray(
                                    payload_decoded_file,
                                    dtype=np.float32,
                                ).reshape(-1)
                            ),
                        )

                if algorithm == "wavmark":
                    # embedded vs decoded-from-saved-WAV payload bits; 0% = perfect bit recovery — use SNR for waveform change
                    print(
                        f"  BER: {ber_val:.3f}%"
                    )
                else:
                    print(f"  BER (detector proxy): {ber_val:.4f}")
                print(f"  NC: {nc_val:.3f}")

                # ------- End of metrics (WavMark BER reflects on-disk WAV) -------

                if watermark_metrics_csv_path is not None:
                    with open(watermark_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
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

                # Generate plots
                if not args.no_plots and plot_dir is not None:
                    plot_path = os.path.join(plot_dir, f"{stem}_{algorithm}_original_vs_watermarked.png")
                    save_original_vs_watermarked_plot(
                        metrics_ref,
                        watermarked_audio,
                        metrics_sr,
                        suptitle=f"{algorithm}: {audio_file} — original vs watermarked",
                        out_path=plot_path,
                        dpi=args.plot_dpi,
                    )
                    #print(f"  Saved plot: {stem}_original_vs_watermarked.png")

            # Detect watermark
            if args.command == "detect":
                det_thresh = args.detection_threshold
                msg_t = args.message_threshold
                frac_t = args.file_fraction_threshold

                if algorithm == "audioseal":
                    # High-level: fraction of frames with P(watermark) > detection_threshold (see AudioSeal AudioSealDetector.detect_watermark).
                    detect_frame_fraction, binary_message = detector.detect_watermark(
                        wav,
                        message_threshold=msg_t,
                        detection_threshold=det_thresh,
                    )
                    frame_logits, message_probs = detector(wav)
                    print_watermark_detection_summary(
                        detect_frame_fraction=detect_frame_fraction,
                        binary_message=binary_message,
                        frame_logits=frame_logits,
                        message_probs=message_probs,
                        detection_threshold=det_thresh,
                        file_fraction_threshold=frac_t,
                        show_raw=bool(args.debug),
                    )
                    log_path = detect_log_by_algorithm.get(algorithm)
                    if log_path is not None:
                        row = detection_log_csv_row_audioseal(
                            audio_file,
                            algorithm,
                            sample_rate=sample_rate,
                            message_threshold=msg_t,
                            detection_threshold=det_thresh,
                            file_fraction_threshold=frac_t,
                            detect_frame_fraction=detect_frame_fraction,
                            binary_message=binary_message,
                            frame_logits=frame_logits,
                            message_probs=message_probs,
                        )
                        with open(log_path, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(row)
                elif algorithm == "wavmark":
                    wm_mono = wavmark_mono_16k_tensor(wav, sample_rate)
                    payload_decoded, decode_info, wav_16k = wavmark_decode_watermark(
                        wmmodel, wm_mono, show_progress=False
                    )
                    print_wavmark_detection_summary(
                        payload_decoded=payload_decoded,
                        decode_info=decode_info,
                        wav_16k=wav_16k,
                        show_raw=bool(args.debug),
                    )
                    log_path = detect_log_by_algorithm.get(algorithm)
                    if log_path is not None:
                        row = detection_log_csv_row_wavmark(
                            audio_file,
                            algorithm,
                            sample_rate=sample_rate,
                            message_threshold=msg_t,
                            detection_threshold=det_thresh,
                            file_fraction_threshold=frac_t,
                            wav_16k=wav_16k,
                            payload_decoded=payload_decoded,
                            decode_info=decode_info,
                        )
                        with open(log_path, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(row)

            # Attack: embed watermark, apply AudioEffects, verify robustness per algorithm.
            if args.command == "attack":
                msg_t = args.message_threshold
                det_t = args.detection_threshold
                frac_t = args.file_fraction_threshold

                wav_wm: torch.Tensor
                atk_sr: int
                specs: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]]

                if algorithm == "audioseal":
                    with torch.no_grad():
                        wav_wm = wav + asmodel.get_watermark(wav)
                    atk_sr = sample_rate
                    specs = attack_eval_specs(sample_rate)
                    base_ok, base_frac = watermark_file_detected(
                        detector,
                        wav_wm,
                        message_threshold=msg_t,
                        detection_threshold=det_t,
                        file_fraction_threshold=frac_t,
                    )
                    status = "detected (resisted)" if base_ok else "NOT detected"
                    print(
                        f"  Baseline (watermarked, no attack): {status}; "
                        f"frame fraction P(wm)>{det_t}: {base_frac:.1%}"
                    )
                elif algorithm == "wavmark":
                    wm_mono = wavmark_mono_16k_tensor(wav, sample_rate)
                    watermarked_np, _ = wm.encode_watermark(
                        wmmodel, wm_mono, payload, show_progress=False
                    )
                    wav_wm = _numpy_1d_to_batch_tensor(watermarked_np)
                    if torch.cuda.is_available():
                        wav_wm = wav_wm.cuda()
                    atk_sr = 16000
                    specs = attack_eval_specs(16000)
                    base_ok = wavmark_payload_survives_attack(
                        wmmodel, payload, wav_wm, atk_sr
                    )
                    status = "payload recovered" if base_ok else "payload NOT recovered"
                    print(
                        f"  Baseline (watermarked, no attack): {status} "
                        f"(WavMark exact 16-bit payload match vs embedded)"
                    )
                else:
                    raise SystemExit(f"Unsupported algorithm for attack: {algorithm}")

                attacked_root = (
                    os.path.abspath(args.save_attacked)
                    if args.save_attacked
                    else None
                )
                stem, _ = os.path.splitext(audio_file)
                attacked_dir = None
                if attacked_root is not None:
                    attacked_dir = os.path.join(attacked_root, stem)
                    os.makedirs(attacked_dir, exist_ok=True)

                attack_row: dict[str, str] = {name: "-" for name in attack_metric_names}
                for atk_idx, (attack_name, attack_fn) in enumerate(specs):
                    torch.manual_seed(args.attack_seed + atk_idx * 10_007)
                    with torch.no_grad():
                        attacked_wm = attack_fn(wav_wm)
                    attacked_wm = attacked_wm.contiguous()

                    if algorithm == "audioseal":
                        resisted, frac_wm = watermark_file_detected(
                            detector,
                            attacked_wm,
                            message_threshold=msg_t,
                            detection_threshold=det_t,
                            file_fraction_threshold=frac_t,
                        )
                        verdict = (
                            "watermark still detected (attack resisted)"
                            if resisted
                            else "watermark NOT detected (attack succeeded)"
                        )
                        if attacked_wm.shape[-1] != wav_wm.shape[-1]:
                            print(
                                f"  Attack {attack_name!r}: {verdict}; "
                                f"P(wm)>{det_t} frame fraction={frac_wm:.1%}; "
                                f"[length {wav_wm.shape[-1]} -> {attacked_wm.shape[-1]} samples]"
                            )
                        else:
                            print(
                                f"  Attack {attack_name!r}: {verdict}; "
                                f"P(wm)>{det_t} frame fraction={frac_wm:.1%}"
                            )
                    elif algorithm == "wavmark":
                        resisted = wavmark_payload_survives_attack(
                            wmmodel, payload, attacked_wm, atk_sr
                        )
                        verdict = (
                            "payload still recovered (attack resisted)"
                            if resisted
                            else "payload NOT recovered (attack succeeded)"
                        )
                        if attacked_wm.shape[-1] != wav_wm.shape[-1]:
                            print(
                                f"  Attack {attack_name!r}: {verdict}; "
                                f"[length {wav_wm.shape[-1]} -> {attacked_wm.shape[-1]} samples]"
                            )
                        else:
                            print(f"  Attack {attack_name!r}: {verdict}")

                    attack_row[attack_name] = "X" if resisted else "-"

                    if attacked_dir is not None:
                        out_wav = os.path.join(attacked_dir, f"{stem}_{attack_name}.wav")
                        save_watermarked_wav(attacked_wm, atk_sr, out_wav)

                if attack_metrics_csv_path is not None:
                    with open(attack_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [audio_file, algorithm, *[attack_row[name] for name in attack_metric_names]]
                        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping script.", file=sys.stderr, flush=True)
        plt.close("all")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(130)
