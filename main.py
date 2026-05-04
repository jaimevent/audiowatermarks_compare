#!/usr/bin/python

import os

# AudioSeal wraps the encoder with torch.compile; Inductor on Windows needs MSVC (cl.exe).
# Without Build Tools, compilation fails. Disable Dynamo before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
import torch

torch._dynamo.config.disable = True  # belt-and-suspenders if env above is ignored
torch.cuda.empty_cache()

import numpy as np
import soundfile as sf

from algorithms import ALGORITHM_IDS, ALGORITHM_REGISTRY, get_backend
from metrics_csv import (
    append_attack_metrics_row,
    append_watermark_metrics_row,
    metrics_csv_filepath,
    write_attack_metrics_header,
    write_detection_log_header,
    write_watermark_metrics_header,
)
from pesq import pesq
import librosa

from attacks import attack_eval_specs
from watermark_plots import (
    attack_row_values_to_binary,
    close_all_figures,
    save_attack_summary_bar_chart,
    save_attack_summary_heatmap,
    save_original_vs_watermarked_plot,
    waveform_to_mono_numpy,
)


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
    ref = waveform_to_mono_numpy(wav_original[0]).astype(np.float32, copy=False)
    deg = waveform_to_mono_numpy(wav_watermarked[0]).astype(np.float32, copy=False)

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

    ref = waveform_to_mono_numpy(wav_original[0]).astype(np.float32, copy=False)
    deg = waveform_to_mono_numpy(wav_watermarked[0]).astype(np.float32, copy=False)

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

def _resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D numpy audio signal to the target sample rate."""
    if original_sr == target_sr:
        return audio

    resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return np.asarray(resampled, dtype=np.float32)


def _load_hf_token_from_path(path: str) -> bool:
    """If ``path`` exists, read the first line as token and set Hub env vars.

    Returns True if the token was applied. Never logs the token.
    Missing file, read errors, or empty/comment first line → False (caller may fall back).
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            token = f.readline().strip()
    except OSError:
        return False
    if not token or token.startswith("#"):
        return False
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return True


def _configure_hf_token_from_repo_file() -> None:
    """Prefer ``.hf_token`` or ``hf_token.txt`` beside ``main.py``; else leave env unchanged.

    If neither file yields a token, ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` already set in
    the environment (or absent) are used by ``huggingface_hub`` as usual.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for name in (".hf_token", "hf_token.txt"):
        if _load_hf_token_from_path(os.path.join(repo_root, name)):
            return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio watermarking CLI (registered algorithms). "
            "Subcommands: watermark, detect, attack."
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
    parser.add_argument(
        "--silentcipher-model",
        default="44.1k",
        choices=("44.1k", "16k"),
        metavar="TYPE",
        help=(
            "SilentCipher checkpoint family (44.1 kHz vs 16 kHz model; "
            "default: 44.1k). Ignored unless algorithm is silentcipher."
        ),
    )
    parser.add_argument(
        "--silentcipher-phase-shift",
        action="store_true",
        dest="silentcipher_phase_shift",
        help=(
            "SilentCipher decode_wav(..., phase_shift_decoding=True) for watermark/detect: "
            "slower, more robust to crops. The attack command always uses phase-shift decode "
            "for SilentCipher. Ignored unless algorithm is silentcipher."
        ),
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
            choices=ALGORITHM_IDS,
            metavar="ALGORITHM",
            help="Watermark algorithm id (default: %(default)s).",
        )
        p.add_argument(
            "--all-algorithms",
            action="store_true",
            dest="run_all_algorithms",
            help="Run every algorithm registered in algorithms.ALGORITHM_REGISTRY.",
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
    p_attack.add_argument(
        "--output-attack-plot",
        "-o",
        default=None,
        metavar="DIR",
        help=(
            "Directory for attack robustness summary PNGs (heatmap + bar chart). "
            "Default: plots/attack_summary (relative to current working directory)."
        ),
    )
    p_attack.add_argument(
        "--no-attack-plots",
        action="store_true",
        help="Skip attack robustness summary heatmap and bar chart.",
    )
    p_attack.add_argument(
        "--attack-plot-dpi",
        type=int,
        default=150,
        metavar="N",
        help="Resolution of attack summary PNGs (default: 150).",
    )
    attach_algorithm_arguments(p_attack)

    return parser.parse_args()

def main() -> None:
    _configure_hf_token_from_repo_file()
    args = _parse_args()

    audio_folder = os.path.abspath(args.input)
    if not os.path.isdir(audio_folder):
        raise SystemExit(f"Not a directory: {audio_folder}")

    if getattr(args, "run_all_algorithms", False):
        algorithms = sorted(ALGORITHM_REGISTRY.keys())
    else:
        algorithms = [args.algorithm]

    print(f" -- Using the following algorithms: {algorithms} --")

    plot_dir = None
    watermark_metrics_by_algorithm: dict[str, str] = {}

    if args.command == "watermark" and not args.no_plots:
        plot_dir = os.path.abspath(args.output_plot or os.path.join("plots"))
        os.makedirs(plot_dir, exist_ok=True)

    if args.command == "watermark":
        for algorithm in algorithms:
            watermarked_wav_dir = os.path.abspath(
                args.output_watermarked or os.path.join(audio_folder, f"{algorithm}_watermarked_wav")
            )
            os.makedirs(watermarked_wav_dir, exist_ok=True)

            wm_path = metrics_csv_filepath(
                args.output_metrics,
                default_filename=f"{algorithm}_watermark_metrics.csv",
            )
            watermark_metrics_by_algorithm[algorithm] = wm_path
            write_watermark_metrics_header(wm_path)
            print(f"{algorithm} watermark metrics CSV: {wm_path}")

    attack_metric_names: list[str] = []
    attack_metrics_by_algorithm: dict[str, str] = {}
    if args.command == "attack":
        for algorithm in algorithms:
            attack_metric_names = [name for name, _ in attack_eval_specs(sample_rate=16000)]
            am_path = metrics_csv_filepath(
                args.output_attack_metrics,
                default_filename=f"{algorithm}_attack_metrics.csv",
            )
            attack_metrics_by_algorithm[algorithm] = am_path
            write_attack_metrics_header(am_path, attack_metric_names)
            print(f"{algorithm} attack metrics CSV: {am_path}")

    attack_plot_dir: str | None = None
    if args.command == "attack" and not args.no_attack_plots:
        attack_plot_dir = os.path.abspath(args.output_attack_plot or os.path.join("plots", "attack_summary"))
        os.makedirs(attack_plot_dir, exist_ok=True)
        print(f"Attack robustness plots (heatmap + bars): {attack_plot_dir}")

    detect_log_by_algorithm: dict[str, str] = {}
    if args.command == "detect":
        for algorithm in algorithms:
            path = metrics_csv_filepath(
                args.output_detect_log,
                default_filename=f"{algorithm}_detection_log.csv",
            )
            detect_log_by_algorithm[algorithm] = path
            write_detection_log_header(path)
            print(f"{algorithm} detection log CSV: {path}")

    audio_files = [
        f
        for f in os.listdir(audio_folder)
        if f.lower().endswith(".wav") or f.lower().endswith(".flac")
    ]

    for algorithm in algorithms:
        backend = get_backend(algorithm, args)
        backend.setup(args.command)

        print(f"You've selected the following command: {args.command}")

        attack_resistance_rows: list[list[int]] = []
        attack_resistance_labels: list[str] = []

        for audio_file in audio_files:
            print(f"---- Processing audio file: {audio_file} for {algorithm} ----")
            audio_path = os.path.join(audio_folder, audio_file)
            wav, sample_rate = load_audio(audio_path)

            # Skip if audio is too long to avoid OOM
            max_samples = 16000 * 60  # 1 minutes at 16 kHz
            if wav.shape[-1] > max_samples:
                print(f"  Audio file {audio_file} is too long ({wav.shape[-1] / sample_rate:.1f} seconds), skipping.")
                continue
            
            wav = backend.prepare_wav_tensor(wav)
            wav = wav.unsqueeze(0)

            if args.command == "watermark":
                start = time.perf_counter()
                watermarked_wav_dir = os.path.abspath(
                    args.output_watermarked or os.path.join(audio_folder, f"{algorithm}_watermarked_wav")
                )
                metrics_ref, watermarked_audio, metrics_sr = backend.embed_watermark(wav, sample_rate)
                watermarked_audio = backend.finalize_watermarked_cuda(watermarked_audio)

                snr_db = watermarking_snr_db(metrics_ref, watermarked_audio)
                print(f"  SNR: {snr_db:.2f} dB")

                pesq_val = pesq_score(metrics_ref, watermarked_audio, metrics_sr)
                print(f"  PESQ: {pesq_val:.2f}")

                ber_val, nc_val = 0.0, 0.0
                pre = backend.compute_ber_nc_before_save(wav, watermarked_audio, sample_rate)
                if pre is not None:
                    ber_val, nc_val = pre

                stem, _ = os.path.splitext(audio_file)
                wav_out_path = os.path.join(
                    watermarked_wav_dir, f"{stem}_{algorithm}_watermarked.wav"
                )
                save_watermarked_wav(watermarked_audio, metrics_sr, wav_out_path)
                print(f"  Saved watermarked WAV: {stem}_{algorithm}_watermarked.wav")

                post = backend.compute_ber_nc_after_save(wav_out_path)
                if post is not None:
                    ber_val, nc_val = post

                backend.print_ber_line(ber_val)
                print(f"  NC: {nc_val:.3f}")

                if not args.no_plots and plot_dir is not None:
                    plot_path = os.path.join(
                        plot_dir, f"{stem}_{algorithm}_original_vs_watermarked.png"
                    )

                    save_original_vs_watermarked_plot(
                        metrics_ref,
                        watermarked_audio,
                        metrics_sr,
                        suptitle=f"{algorithm}: {audio_file} — original vs watermarked",
                        out_path=plot_path,
                        dpi=args.plot_dpi,
                    )

                processing_ms = (time.perf_counter() - start) * 1000.0
                wm_csv = watermark_metrics_by_algorithm.get(algorithm)
                if wm_csv is not None:
                    append_watermark_metrics_row(
                        wm_csv,
                        audio_file=audio_file,
                        algorithm=algorithm,
                        processing_ms=processing_ms,
                        snr_db=snr_db,
                        pesq_val=pesq_val,
                        ber_val=ber_val,
                        nc_val=nc_val,
                    )

                # Free GPU memory for watermark command
                del metrics_ref
                del watermarked_audio

            if args.command == "detect":
                backend.run_detect(
                    wav,
                    sample_rate,
                    audio_file,
                    detect_log_by_algorithm.get(algorithm),
                )

            if args.command == "attack":
                start = time.perf_counter()
                msg_t = args.message_threshold
                det_t = args.detection_threshold
                frac_t = args.file_fraction_threshold

                wav_wm, atk_sr = backend.attack_prepare(wav, sample_rate)
                specs = attack_eval_specs(atk_sr)
                backend.attack_print_baseline(wav_wm, atk_sr, msg_t, det_t, frac_t)

                attacked_root = (
                    os.path.abspath(args.save_attacked) if args.save_attacked else None
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

                    resisted = backend.attack_evaluate(
                        wav_wm,
                        attacked_wm,
                        atk_sr,
                        attack_name,
                        msg_t,
                        det_t,
                        frac_t,
                    )
                    attack_row[attack_name] = "X" if resisted else "-"

                    if attacked_dir is not None:
                        out_wav = os.path.join(attacked_dir, f"{stem}_{attack_name}.wav")
                        save_watermarked_wav(attacked_wm, atk_sr, out_wav)

                    # Free memory after each attack
                    del attacked_wm

                processing_ms = (time.perf_counter() - start) * 1000.0
                attack_csv = attack_metrics_by_algorithm.get(algorithm)
                if attack_csv is not None:
                    append_attack_metrics_row(
                        attack_csv,
                        audio_file=audio_file,
                        algorithm=algorithm,
                        processing_ms=processing_ms,
                        attack_metric_names=attack_metric_names,
                        attack_row=attack_row,
                    )
                attack_resistance_rows.append(
                    attack_row_values_to_binary(attack_row, attack_metric_names)
                )
                attack_resistance_labels.append(audio_file)

                # Free GPU memory for attack command
                del wav_wm

            # Free common GPU memory
            del wav
            torch.cuda.empty_cache()

        # Free backend and models after processing all files for this algorithm
        del backend
        torch.cuda.empty_cache()

        if (
            args.command == "attack"
            and attack_plot_dir is not None
            and attack_metric_names
            and attack_resistance_rows
        ):
            heatmap_path = os.path.join(attack_plot_dir, f"{algorithm}_attack_resistance_heatmap.png")
            bars_path = os.path.join(attack_plot_dir, f"{algorithm}_attack_resistance_bars.png")
            save_attack_summary_heatmap(
                np.array(attack_resistance_rows, dtype=np.float64),
                attack_resistance_labels,
                attack_metric_names,
                algorithm=algorithm,
                out_path=heatmap_path,
                dpi=int(args.attack_plot_dpi),
            )
            save_attack_summary_bar_chart(
                np.array(attack_resistance_rows, dtype=np.float64),
                attack_metric_names,
                algorithm=algorithm,
                out_path=bars_path,
                dpi=int(args.attack_plot_dpi),
            )
            print(f"  Saved attack summary PNGs: {heatmap_path}, {bars_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping script.", file=sys.stderr, flush=True)
        close_all_figures()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(130)
