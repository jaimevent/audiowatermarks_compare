"""WavMark I/O helpers: mono 16 kHz prep and decoding layout matching the wavmark library."""

from __future__ import annotations

import numpy as np
import torch

# Same layout as wavmark.utils.wm_decode_util.extract_watermark_v3_batch (shift 0.1, p 0.5).
DECODE_NUM_SAMPLES = 16000
DECODE_SHIFT_STEP = int(0.1 * DECODE_NUM_SAMPLES * 0.5)


def resample_audio_mono(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    out = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return np.asarray(out, dtype=np.float32)


def wavmark_mono_16k_tensor(wav: torch.Tensor, sample_rate: int) -> np.ndarray:
    """Mono float32 waveform at 16 kHz; ``wav`` shaped (batch, ch, samp) or (ch, samp)."""
    x = wav.detach().cpu()
    if x.dim() == 3:
        x = x[0]
    if x.dim() != 2:
        raise ValueError(
            "Expected wav (batch, channels, samples) or (channels, samples); "
            f"got shape {tuple(wav.shape)}"
        )
    mono = x.mean(dim=0).numpy().astype(np.float32, copy=False)
    return resample_audio_mono(mono, sample_rate, 16000)


def wavmark_signal_for_decoder(wav_mono_16k: np.ndarray) -> np.ndarray:
    """Pad so upstream sliding decode has ≥1 candidate offset when 16000 ≤ len < 16800."""
    sig = np.asarray(wav_mono_16k, dtype=np.float32, order="C").reshape(-1)
    need_len = DECODE_NUM_SAMPLES + DECODE_SHIFT_STEP
    if len(sig) < DECODE_NUM_SAMPLES:
        return sig
    if len(sig) < need_len:
        sig = np.concatenate([sig, np.zeros(need_len - len(sig), dtype=np.float32)])
    return sig


def wavmark_decode_sliding_stats(wav_16k: np.ndarray, decode_info: dict) -> tuple[int, float, list[float]]:
    shift_step = DECODE_SHIFT_STEP
    num_point = DECODE_NUM_SAMPLES
    total_windows = max(0, (len(wav_16k) - num_point) // shift_step)
    results = decode_info.get("results") or []
    n_sync = len(results)
    if total_windows > 0:
        sync_frac = n_sync / total_windows
    else:
        sync_frac = 1.0 if n_sync > 0 else 0.0
    sims = [float(r["sim"]) for r in results]
    return total_windows, float(sync_frac), sims


def wavmark_decode_watermark(
    model, wav_mono_16k: np.ndarray, *, show_progress: bool = False
) -> tuple[np.ndarray | None, dict, np.ndarray]:
    """Decode WavMark payload; returns ``(payload | None, info, padded_1d_signal)`` for logging/stats."""
    import wavmark as wm

    sig = wavmark_signal_for_decoder(wav_mono_16k)
    payload, info = wm.decode_watermark(model, sig, show_progress=show_progress)
    return payload, info, sig
