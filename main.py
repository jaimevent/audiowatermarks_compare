#!/usr/bin/python

import os

# AudioSeal wraps the encoder with torch.compile; Inductor on Windows needs MSVC (cl.exe).
# Without Build Tools, compilation fails. Disable Dynamo before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import re
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pesq import pesq
import librosa

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
    p_watermark.add_argument(
        "--output-plot",
        "-o",
        default=None,
        metavar="DIR",
        help="Directory for PNG plots (default: <input>/audioseal_plots).",
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

    p_attack = subparsers.add_parser(
        "attack",
        help="Apply attacks and evaluate detection.",
    )
    p_attack.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="DIR",
        help=input_help,
    )

    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    audio_folder = os.path.abspath(args.input)
    if not os.path.isdir(audio_folder):
        raise SystemExit(f"Not a directory: {audio_folder}")

    plot_dir = None
    if args.command == "watermark" and not args.no_plots:
        plot_dir = os.path.abspath(
            args.output_plot or os.path.join(audio_folder, "audioseal_plots")
        )
        os.makedirs(plot_dir, exist_ok=True)

    if args.command == "watermark":
        watermarked_wav_dir = os.path.abspath(
            args.output_watermarked or os.path.join(audio_folder, "watermarked_wav")
        )
        os.makedirs(watermarked_wav_dir, exist_ok=True)

    model = AudioSeal.load_generator(args.generator)
    model.eval()
    detector = AudioSeal.load_detector(args.detector)
    detector.eval()

    # Move models to GPU if available (tensors must match detector device).
    if torch.cuda.is_available():
        model = model.cuda()
        detector = detector.cuda()

    print(f"You've selected the following command: {args.command}")

    audio_files = [
        f
        for f in os.listdir(audio_folder)
        if f.lower().endswith(".wav") or f.lower().endswith(".flac")
    ]
    for audio_file in audio_files:
        print(f"---- Processing audio file: {audio_file} ----")
        audio_path = os.path.join(audio_folder, audio_file)
        wav, sample_rate = load_audio(audio_path)
        # Move model to GPU if available
        if torch.cuda.is_available():
            wav = wav.cuda()

        # load_audio returns wav shaped (channels, samples) (from soundfile: channels first). After wav.unsqueeze(0) it returns 
        # (1, channels, samples): batch size 1, which matches what AudioSeal expects (batch × channels × samples).
        wav = wav.unsqueeze(0)

        # Add watermark
        if args.command == "watermark":            
            watermark = model.get_watermark(wav)
            watermarked_audio = wav + watermark

            if torch.cuda.is_available():
                watermarked_audio = watermarked_audio.cuda()
            
            # ------- Start of metrics -------
            # Measure SNR
            snr_db = watermarking_snr_db(wav, watermarked_audio)
            print(f"  SNR (original vs residual): {snr_db:.2f} dB")

            # Measure PESQ
            pesq_val = pesq_score(wav, watermarked_audio, sample_rate)
            print(f"  PESQ score: {pesq_val:.2f}")

            # Measure BER as decoded-bit mismatch rate (original vs watermarked).
            # This is a detector-based proxy BER when explicit payload ground-truth
            # bits are not provided by the embedding call.
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
            print(f"  BER (decoded bits mismatch rate): {ber_val:.4f}")
            nc_val = normalized_correlation(bits_original, bits_watermarked)
            print(f"  NC (normalized correlation): {nc_val:.4f}")

            # Measure ODG (requires external PEAQ backend in PATH).
            # try:
            #    odg_val = odg_score(wav, watermarked_audio, sample_rate)
            #    print(f"  ODG score: {odg_val:.2f}")
            #except RuntimeError as exc:
            #    print(f"  ODG score: unavailable ({exc})")
            
            # ------- End of metrics -------

            # Save watermarked audio
            stem, _ = os.path.splitext(audio_file)
            wav_out_path = os.path.join(watermarked_wav_dir, f"{stem}_watermarked.wav")
            save_watermarked_wav(watermarked_audio, sample_rate, wav_out_path)
            print(f"  Saved watermarked WAV: {stem}_watermarked.wav")

            # Generate plots
            if not args.no_plots and plot_dir is not None:
                plot_path = os.path.join(plot_dir, f"{stem}_original_vs_watermarked.png")
                save_original_vs_watermarked_plot(
                    wav,
                    watermarked_audio,
                    sample_rate,
                    suptitle=f"{audio_file} — original vs watermarked",
                    out_path=plot_path,
                    dpi=args.plot_dpi,
                )
                print(f"  Saved plot: {stem}_original_vs_watermarked.png")

        # Detect watermark
        if args.command == "detect":
            # High-level: fraction of frames with P(watermark) > detection_threshold (see AudioSeal AudioSealDetector.detect_watermark).
            det_thresh = 0.5
            detect_frame_fraction, binary_message = detector.detect_watermark(
                wav,
                message_threshold=0.5,
                detection_threshold=det_thresh,
            )
            frame_logits, message_probs = detector(wav)
            print_watermark_detection_summary(
                detect_frame_fraction=detect_frame_fraction,
                binary_message=binary_message,
                frame_logits=frame_logits,
                message_probs=message_probs,
                detection_threshold=det_thresh,
                file_fraction_threshold=0.5,
                show_raw=bool(args.debug),
            )

        # Attack
        if args.command == "attack":
            pass
            return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping script.", file=sys.stderr, flush=True)
        plt.close("all")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(130)
