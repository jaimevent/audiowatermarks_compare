#!/usr/bin/python

import os

# AudioSeal wraps the encoder with torch.compile; Inductor on Windows needs MSVC (cl.exe).
# Without Build Tools, compilation fails. Disable Dynamo before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import sys
import tempfile
import torch

torch._dynamo.config.disable = True  # belt-and-suspenders if env above is ignored

import numpy as np
import soundfile as sf
from audioseal import AudioSeal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def convert_flac_to_wav(src_flac: str, dst_wav: str) -> None:
    """Decode FLAC and write a WAV container (float32 samples). libsndfile handles FLAC."""
    data, samplerate = sf.read(src_flac, dtype="float32", always_2d=True)
    sf.write(dst_wav, data, samplerate, format="WAV", subtype="FLOAT")


def _load_from_wav_file(wav_path: str) -> tuple[torch.Tensor, int]:
    # soundfile avoids torchaudio 2.10+ routing through torchcodec (FFmpeg DLLs on Windows).
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AudioSeal on each .wav / .flac in a folder: embed watermark, detect, "
            "and optionally save original vs watermarked comparison plots."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="DIR",
        help="Directory containing .wav and/or .flac files (non-recursive).",
    )
    parser.add_argument(
        "--output-plot",
        "-o",
        default=None,
        metavar="DIR",
        help="Directory for PNG plots (default: <input>/audioseal_plots).",
    )
    parser.add_argument(
        "--debug",
        default=True,
        help="Show debug information.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing comparison plots.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        metavar="N",
        help="Resolution of saved PNG figures (default: 150).",
    )
    parser.add_argument(
        "--output-watermarked",
        default=None,
        metavar="DIR",
        help="Directory for watermarked WAV files (default: <input>/watermarked_wav).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audio_folder = os.path.abspath(args.input)
    if not os.path.isdir(audio_folder):
        raise SystemExit(f"Not a directory: {audio_folder}")

    plot_dir = None
    if not args.no_plots:
        plot_dir = os.path.abspath(
            args.output_plot or os.path.join(audio_folder, "audioseal_plots")
        )
        os.makedirs(plot_dir, exist_ok=True)

    watermarked_wav_dir = os.path.abspath(
        args.output_watermarked or os.path.join(audio_folder, "watermarked_wav")
    )
    os.makedirs(watermarked_wav_dir, exist_ok=True)

    model = AudioSeal.load_generator(args.generator)
    model.eval()
    detector = AudioSeal.load_detector(args.detector)
    detector.eval()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    audio_files = [
        f
        for f in os.listdir(audio_folder)
        if f.lower().endswith(".wav") or f.lower().endswith(".flac")
    ]
    for audio_file in audio_files:
        audio_path = os.path.join(audio_folder, audio_file)
        wav, sample_rate = load_audio(audio_path)
        # Move model to GPU if available
        if torch.cuda.is_available():
            wav = wav.cuda()

        wav = wav.unsqueeze(0)
        watermark = model.get_watermark(wav)
        watermarked_audio = wav + watermark

        snr_db = watermarking_snr_db(wav, watermarked_audio)
        print(f"  Watermark SNR (original vs residual): {snr_db:.2f} dB")

        stem, _ = os.path.splitext(audio_file)
        wav_out_path = os.path.join(watermarked_wav_dir, f"{stem}_watermarked.wav")
        save_watermarked_wav(watermarked_audio, sample_rate, wav_out_path)
        print(f"  Saved watermarked WAV: {stem}_watermarked.wav")

        if torch.cuda.is_available():
            watermarked_audio = watermarked_audio.cuda()

        # To detect the messages in the high-level.
        # message_threshold indicates the threshold in which the detector will convert the stochastic messages (with probability 
        # between 0 and 1) into the n-bit binary format. In most of the case, the generator generates an unbiased message from 
        # the secret, so 0.5 is a reasonable choice (so the value > 0.5 means 1 and value < 0.5 means 0).
        result, message = detector.detect_watermark(watermarked_audio, message_threshold=0.5)
        if args.debug:
            print(f"Detect messages in the high-level: Audio: {audio_file}, Result: {result}, Message: {message}")

        # To detect the messages in the low-level.
        result, message = detector(watermarked_audio)
        if args.debug:
            # result is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
            # A watermarked audio should have result[:, 1, :] > 0.5
            print(f"Detect messages in the low-level: Audio: {audio_file}, Result: {result[:, 1 , :]}, Message: {message}")  

        # Generate plots
        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, f"{stem}_original_vs_watermarked.png")
            save_original_vs_watermarked_plot(
                wav,
                watermarked_audio,
                sample_rate,
                suptitle=f"{audio_file} — original vs watermarked",
                out_path=plot_path,
                dpi=args.plot_dpi,
            )
            print(f"  Saved plot: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping script.", file=sys.stderr, flush=True)
        plt.close("all")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(130)

# Other way is to load directly from the checkpoint
# model =  Watermarker.from_pretrained(checkpoint_path, device = wav.device)

# a torch tensor of shape (batch, channels, samples) and a sample rate
# It is important to process the audio to the same sample rate as the model
# expects. The default AudioSeal should work well with 16kHz and 24kHz, and 
# in the case of 48 khZ, it should work well for most speech audios
# wav = [load audio wav into a tensor of BatchxChannelxTime]

# watermark = model.get_watermark(wav)

# Optional: you can add a 16-bit message to embed in the watermark
# msg = torch.randint(0, 2, (wav.shape(0), model.msg_processor.nbits), device=wav.device)
# watermark = model.get_watermark(wav, message = msg)

# watermarked_audio = wav + watermark

# detector = AudioSeal.load_detector("audioseal_detector_16bits")

# To detect the messages in the high-level.
# result, message = detector.detect_watermark(watermarked_audio)

# print(result) # result is a float number indicating the probability of the audio being watermarked,
# print(message)  # message is a binary vector of 16 bits

# To detect the messages in the low-level.
# result, message = detector(watermarked_audio)

# result is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
# A watermarked audio should have result[:, 1, :] > 0.5
# print(result[:, 1 , :])  

# Message is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
# message will be a random tensor if the detector detects no watermarking from the audio
# print(message)