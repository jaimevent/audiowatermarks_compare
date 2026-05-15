"""DSSS spread-spectrum watermark backend for ``main.py``."""

from __future__ import annotations

import csv
import time
from typing import ClassVar

import numpy as np
import torch

from audio_io import load_waveform_torch
from bit_metrics import bit_error_rate, normalized_correlation

from .base import WatermarkBackend

DEFAULT_MESSAGE = "unimilano"
DEFAULT_FRAME_LENGTH = 4096
DEFAULT_ALPHA = 0.01
DEFAULT_SEED = 42


def generate_pn_sequence(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=length).astype(np.float32)


def payload_min_samples(bits: np.ndarray, frame_length: int) -> int:
    """Minimum mono samples required to hold one full payload."""
    return len(bits) * frame_length


def message_to_bits(message: str) -> np.ndarray:
    data = message.encode("utf-8")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.int8)


def bits_to_message(bits: np.ndarray) -> str:
    n_bytes = len(bits) // 8
    if n_bytes == 0:
        return ""
    packed = np.packbits(bits[: n_bytes * 8].astype(np.uint8))
    return packed.tobytes().decode("utf-8", errors="replace").rstrip("\x00")


def pad_audio_with_zeros(audio: np.ndarray, min_samples: int) -> tuple[np.ndarray, int]:
    """Append trailing zeros so ``len(audio) >= min_samples``; return (audio, pad_count)."""
    if len(audio) >= min_samples:
        return audio, 0
    pad_count = min_samples - len(audio)
    padded = np.pad(audio, (0, pad_count), mode="constant", constant_values=0.0)
    return padded.astype(np.float32, copy=False), pad_count


def embed_watermark(
    audio: np.ndarray,
    bits: np.ndarray,
    *,
    alpha: float,
    frame_length: int,
    seed: int,
) -> np.ndarray:
    """Spread-spectrum embed: one PN-modulated bit per frame."""
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if len(bits) == 0:
        raise ValueError("empty payload")
    n_frames = (len(audio) + frame_length - 1) // frame_length
    out = audio.copy()
    for i in range(n_frames):
        start = i * frame_length
        end = min(start + frame_length, len(out))
        seg_len = end - start
        bit = bits[i % len(bits)]
        chip = 1.0 if bit else -1.0
        pn = generate_pn_sequence(seg_len, seed + i)
        out[start:end] = out[start:end] + alpha * chip * pn
    return out


def extract_watermark(
    audio: np.ndarray,
    n_bits: int,
    *,
    frame_length: int,
    seed: int,
) -> np.ndarray:
    """Spread-spectrum extract: correlate each frame with its PN sequence."""
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        start = i * frame_length
        end = min(start + frame_length, len(audio))
        if start >= len(audio):
            bits[i] = 0
            continue
        segment = audio[start:end]
        pn = generate_pn_sequence(len(segment), seed + i)
        correlation = float(np.dot(segment, pn))
        bits[i] = 1 if correlation > 0 else 0
    return bits


def wav_batch_to_mono_numpy(wav_batched: torch.Tensor) -> np.ndarray:
    x = wav_batched[0].detach().cpu().numpy()
    if x.ndim == 1:
        return np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return np.mean(x, axis=0).astype(np.float32)
    raise ValueError(f"Unexpected waveform shape after batch[0]: {x.shape}")


def numpy_1d_to_batch_tensor(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).unsqueeze(0)


def payload_ber_nc(expected: np.ndarray, recovered: np.ndarray) -> tuple[float, float]:
    exp = torch.from_numpy(expected.astype(np.float32))
    rec = torch.from_numpy(recovered.astype(np.float32))
    return bit_error_rate(exp, rec), normalized_correlation(exp, rec)


class DsssBackend(WatermarkBackend):
    name: ClassVar[str] = "dsss"

    def setup(self, command: str) -> None:
        args = self.args
        self._message = getattr(args, "dsss_message", DEFAULT_MESSAGE)
        self._frame_length = int(getattr(args, "dsss_frame_length", DEFAULT_FRAME_LENGTH))
        self._alpha = float(getattr(args, "dsss_alpha", DEFAULT_ALPHA))
        self._seed = int(getattr(args, "dsss_seed", DEFAULT_SEED))
        self._bits = message_to_bits(self._message)
        self._min_samples = payload_min_samples(self._bits, self._frame_length)
        if command in ("watermark", "attack", "detect"):
            print(
                f"DSSS payload {self._message!r} ({len(self._bits)} bits), "
                f"frame_length={self._frame_length}, alpha={self._alpha}, seed={self._seed}"
            )

    def prepare_wav_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[0] > 1:
            return wav.mean(dim=0, keepdim=True)
        return wav

    def _embed_mono(self, y: np.ndarray) -> np.ndarray:
        return embed_watermark(
            y,
            self._bits,
            alpha=self._alpha,
            frame_length=self._frame_length,
            seed=self._seed,
        )

    def _extract_bits(self, y: np.ndarray) -> np.ndarray:
        return extract_watermark(
            y,
            len(self._bits),
            frame_length=self._frame_length,
            seed=self._seed,
        )

    def embed_watermark(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        y = wav_batch_to_mono_numpy(wav_batched)
        y, _ = pad_audio_with_zeros(y, self._min_samples)
        watermarked = self._embed_mono(y)
        metrics_ref = numpy_1d_to_batch_tensor(y)
        wm = numpy_1d_to_batch_tensor(watermarked)
        return metrics_ref, wm, int(sample_rate)

    def finalize_watermarked_cuda(self, watermarked: torch.Tensor) -> torch.Tensor:
        return watermarked

    def _ber_nc_from_mono(self, y: np.ndarray) -> tuple[float, float]:
        recovered = self._extract_bits(y)
        return payload_ber_nc(self._bits, recovered)

    def compute_ber_nc_before_save(
        self,
        wav_batched: torch.Tensor,
        watermarked: torch.Tensor,
        sample_rate: int,
    ) -> tuple[float, float] | None:
        del wav_batched, watermarked, sample_rate
        return None

    def compute_ber_nc_after_save(self, wav_out_path: str) -> tuple[float, float] | None:
        wav, _sr = load_waveform_torch(wav_out_path)
        y = wav_batch_to_mono_numpy(wav.unsqueeze(0))
        return self._ber_nc_from_mono(y)

    def print_ber_line(self, ber_val: float) -> None:
        print(f"  BER (payload bits): {ber_val:.4f} ({ber_val * 100:.2f}%)")

    def run_detect(
        self,
        wav_batched: torch.Tensor,
        sample_rate: int,
        audio_file: str,
        detect_log_path: str | None,
        elapsed_ms: float | None = None,
    ) -> None:
        start = time.perf_counter()
        y = wav_batch_to_mono_numpy(wav_batched)
        recovered = self._extract_bits(y)
        decoded = bits_to_message(recovered)
        ber, nc = payload_ber_nc(self._bits, recovered)
        ok = decoded == self._message
        verdict = "payload recovered" if ok else "payload NOT recovered"
        print(f"  DSSS: {verdict} (decoded={decoded!r}, reference={self._message!r})")
        print(f"  BER: {ber:.4f} ({ber * 100:.2f}%), NC: {nc:.3f}")
        if detect_log_path is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            bit_str = "".join(str(int(b)) for b in recovered.tolist())
            row = [
                audio_file,
                self.name,
                "X" if ok else "-",
                sample_rate,
                self.args.detection_threshold,
                self.args.file_fraction_threshold,
                self.args.message_threshold,
                round(1.0 if ok else 0.0, 6),
                1,
                round(1.0 - ber, 6),
                round(1.0 if ok else 0.0, 6),
                0.0,
                round(nc, 6),
                round(nc, 6),
                bit_str,
                decoded,
                round(elapsed_ms, 2),
            ]
            with open(detect_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def attack_prepare(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        y = wav_batch_to_mono_numpy(wav_batched)
        y, _ = pad_audio_with_zeros(y, self._min_samples)
        watermarked = self._embed_mono(y)
        return numpy_1d_to_batch_tensor(watermarked), int(sample_rate)

    def attack_print_baseline(
        self,
        wav_wm: torch.Tensor,
        atk_sr: int,
        msg_t: float,
        det_t: float,
        frac_t: float,
    ) -> None:
        del msg_t, det_t, frac_t, atk_sr
        y = wav_batch_to_mono_numpy(wav_wm)
        recovered = self._extract_bits(y)
        ok = bits_to_message(recovered) == self._message
        status = "payload recovered" if ok else "payload NOT recovered"
        print(f"  Baseline (watermarked, no attack): {status} (DSSS message match)")

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
        del wav_wm, atk_sr, msg_t, det_t, frac_t
        y = wav_batch_to_mono_numpy(attacked_wm)
        recovered = self._extract_bits(y)
        resisted = bits_to_message(recovered) == self._message
        verdict = (
            "payload still recovered (attack resisted)"
            if resisted
            else "payload NOT recovered (attack succeeded)"
        )
        print(f"  Attack {attack_name!r}: {verdict}")
        return resisted
