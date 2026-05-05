"""Load/save helpers for WAV, FLAC, and MP3 waveforms used by ``main`` and backends."""

from __future__ import annotations

import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch


def load_waveform_torch(file_path: str) -> tuple[torch.Tensor, int]:
    """Load audio as float32 tensor shaped ``(channels, samples)`` and sample rate."""
    lower = file_path.lower()
    if lower.endswith(".mp3"):
        data, sr = librosa.load(file_path, sr=None, mono=False)
        if data.ndim == 1:
            y = np.asarray(data, dtype=np.float32)[np.newaxis, :]
        else:
            y = np.ascontiguousarray(data, dtype=np.float32)
        return torch.from_numpy(y), int(sr)
    if lower.endswith(".flac"):
        arr, sr = sf.read(file_path, dtype="float32", always_2d=True)
        return torch.from_numpy(arr.T.copy()), int(sr)
    arr, sr = sf.read(file_path, dtype="float32", always_2d=True)
    return torch.from_numpy(arr.T.copy()), int(sr)


def save_watermarked_to_path(
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    out_path: str,
) -> None:
    """Write a watermarked batch tensor to ``out_path`` (``.wav``, ``.flac``, or ``.mp3``)."""
    if watermarked_audio.dim() != 3:
        raise ValueError(
            f"Expected watermarked tensor rank 3 (batch, channels, samples); got {watermarked_audio.dim()}"
        )
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".wav", ".flac", ".mp3"):
        raise ValueError(
            f"Unsupported output extension {ext!r}; use .wav, .flac, or .mp3 (or output-format=wav)."
        )
    data_smp_ch = watermarked_audio[0].detach().cpu().numpy().T.astype(np.float32, copy=False)
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    if ext == ".wav":
        sf.write(out_abs, data_smp_ch, sample_rate, format="WAV", subtype="FLOAT")
        return
    if ext == ".flac":
        sf.write(out_abs, data_smp_ch, sample_rate, format="FLAC", subtype="FLOAT")
        return
    _write_mp3(watermarked_audio[0].detach().cpu().float(), sample_rate, out_abs)


def _write_mp3(wav_ch_first: torch.Tensor, sample_rate: int, out_abs: str) -> None:
    torchaudio_err: Exception | None = None
    try:
        import torchaudio

        torchaudio.save(out_abs, wav_ch_first, sample_rate, format="mp3")
        return
    except Exception as e:
        torchaudio_err = e

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        data = wav_ch_first.detach().cpu().numpy().astype(np.float32, copy=False).T
        sf.write(tmp_wav, data, sample_rate, format="WAV", subtype="FLOAT")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                tmp_wav,
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                out_abs,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        hint = (
            "MP3 export needs a working TorchAudio FFmpeg backend or ``ffmpeg`` in PATH "
            "(for WAV→MP3 encoding)."
        )
        raise RuntimeError(hint) from (torchaudio_err or e)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg MP3 encoding failed{f': {stderr}' if stderr else ''}. "
            f"TorchAudio also failed: {torchaudio_err}"
        ) from e
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass
