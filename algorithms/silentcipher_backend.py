"""SilentCipher watermark integration (``encode_wav`` / ``decode_wav``).

See SilentCipher docs: message is **five 8-bit integers** (40 bits total); models are
``44.1k`` or ``16k`` sampling-rate families loaded via ``silentcipher.get_model``.
"""

from __future__ import annotations

import csv
from typing import ClassVar

import numpy as np
import soundfile as sf
import silentcipher
import time
import torch

from bit_metrics import normalized_correlation

from .base import WatermarkBackend


def numpy_1d_to_batch_tensor(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).unsqueeze(0)


def wav_batch_to_mono_numpy(wav_batched: torch.Tensor) -> np.ndarray:
    """``(batch, channels, samples)`` → 1-D mono float32 (channel average)."""
    x = wav_batched[0].detach().cpu().numpy()
    if x.ndim == 1:
        return np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return np.mean(x, axis=0).astype(np.float32)
    raise ValueError(f"Unexpected waveform shape after batch[0]: {x.shape}")


def random_five_byte_message() -> list[int]:
    """Payload required by SilentCipher: five bytes ∈ [0, 255]."""
    return np.random.randint(0, 256, size=5, dtype=np.int64).tolist()


def normalize_decoded_message(msg) -> list[int] | None:
    if msg is None:
        return None
    try:
        return [int(x) for x in msg]
    except (TypeError, ValueError):
        return None


def message_bit_ber_nc(
    embedded: list[int],
    decoded: list[int] | None,
) -> tuple[float, float]:
    """BER (% of 40 bits) and NC between bit vectors."""
    if decoded is None or len(embedded) != 5 or len(decoded) != 5:
        return 100.0, 0.0
    e = np.unpackbits(np.asarray(embedded, dtype=np.uint8))
    d = np.unpackbits(np.asarray(decoded, dtype=np.uint8))
    if e.size != d.size:
        return 100.0, 0.0
    ber = float((e != d).mean() * 100.0)
    nc = normalized_correlation(
        torch.from_numpy(e.astype(np.float32)),
        torch.from_numpy(d.astype(np.float32)),
    )
    return ber, nc


def silentcipher_message_recovered(
    model,
    embedded: list[int],
    wav_batched: torch.Tensor,
    sample_rate: int,
    *,
    phase_shift: bool,
) -> bool:
    """True iff decode reports success and first message matches ``embedded``."""
    y = wav_batch_to_mono_numpy(wav_batched)
    result = model.decode_wav(
        y,
        int(sample_rate),
        phase_shift_decoding=phase_shift,
    )
    if not result.get("status"):
        return False
    msgs = result.get("messages") or []
    if not msgs:
        return False
    got = normalize_decoded_message(msgs[0])
    return got == embedded


def print_silentcipher_detection_summary(*, result: dict, show_raw: bool) -> None:
    status = bool(result.get("status"))
    verdict = "watermark likely present" if status else "watermark likely NOT present"
    print(f"  SilentCipher file verdict: {verdict} (decode status={status})")
    messages = result.get("messages") or []
    confidences = result.get("confidences") or []
    if messages:
        print(f"  Decoded message(s): {messages}")
    if confidences:
        print(f"  Confidences: {confidences}")
    if show_raw:
        print(f"    [debug] decode keys: {list(result.keys())}")


def detection_log_csv_row_silentcipher(
    audio_file: str,
    algorithm: str,
    *,
    sample_rate: int,
    message_threshold: float,
    detection_threshold: float,
    file_fraction_threshold: float,
    result: dict,
    elapsed_ms: float | None = None,
) -> list:
    """Map SilentCipher decode output into the shared detection CSV schema."""
    status = bool(result.get("status"))
    detected = "X" if status else "-"
    messages = result.get("messages") or []
    confidences = result.get("confidences") or []
    msg0 = messages[0] if messages else []
    conf0 = float(confidences[0]) if confidences else float("nan")
    bit_str = ";".join(str(int(x)) for x in msg0) if msg0 else ""

    row = [
        audio_file,
        algorithm,
        detected,
        sample_rate,
        detection_threshold,
        file_fraction_threshold,
        message_threshold,
        round(1.0 if status else 0.0, 6),
        1,
        round(conf0, 6),
        round(1.0 if status else 0.0, 6),
        0.0,
        round(conf0, 6),
        round(conf0, 6),
        bit_str,
        "",
    ]
    if elapsed_ms is not None:
        row.append(round(elapsed_ms, 2))
    return row


class SilentCipherBackend(WatermarkBackend):
    name: ClassVar[str] = "silentcipher"

    def setup(self, command: str) -> None:
        args = self.args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = getattr(args, "silentcipher_model", "44.1k")
        self._model = silentcipher.get_model(model_type=model_type, device=device)
        self._message: list[int] | None = None
        # Attack suite stresses crops/time edits; library recommends phase_shift_decoding=True there.
        self._phase_shift = True if command == "attack" and device == "cuda" else bool(
            getattr(args, "silentcipher_phase_shift", False)
        )
        if command in ("watermark", "attack"):
            self._message = random_five_byte_message()
            print("SilentCipher message (5× uint8):", self._message)

    def prepare_wav_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        return wav

    def embed_watermark(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        assert self._message is not None
        y = wav_batch_to_mono_numpy(wav_batched)
        sr = int(sample_rate)
        # Convert numpy to torch tensor on the same device as the model to avoid device mismatch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        y_tensor = torch.from_numpy(y).to(device)
        # calc_sdr=False: library sdr() mixes np.mean with torch tensors and crashes on recent numpy/torch.
        encoded, _sdr = self._model.encode_wav(y_tensor, sr, self._message, calc_sdr=False)
        # Move encoded back to CPU before converting to numpy
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()
        watermarked = numpy_1d_to_batch_tensor(np.asarray(encoded, dtype=np.float32))
        metrics_ref = numpy_1d_to_batch_tensor(y)
        if torch.cuda.is_available():
            watermarked = watermarked.cuda()
            metrics_ref = metrics_ref.cuda()
        return metrics_ref, watermarked, sr

    def finalize_watermarked_cuda(self, watermarked: torch.Tensor) -> torch.Tensor:
        return watermarked

    def compute_ber_nc_before_save(
        self,
        wav_batched: torch.Tensor,
        watermarked: torch.Tensor,
        sample_rate: int,
    ) -> tuple[float, float] | None:
        return None

    def compute_ber_nc_after_save(self, wav_out_path: str) -> tuple[float, float] | None:
        assert self._message is not None
        data, sr_disk = sf.read(wav_out_path, dtype="float32", always_2d=True)
        y = np.mean(data, axis=1).astype(np.float32)
        result = self._model.decode_wav(
            y,
            int(sr_disk),
            phase_shift_decoding=self._phase_shift,
        )
        msgs = result.get("messages") or []
        dec = normalize_decoded_message(msgs[0]) if msgs else None
        return message_bit_ber_nc(self._message, dec)

    def print_ber_line(self, ber_val: float) -> None:
        print(f"  BER (40-bit payload): {ber_val:.3f}%")

    def run_detect(
        self,
        wav_batched: torch.Tensor,
        sample_rate: int,
        audio_file: str,
        detect_log_path: str | None,
        elapsed_ms: float | None = None,
    ) -> None:
        args = self.args
        start = time.perf_counter()
        y = wav_batch_to_mono_numpy(wav_batched)
        result = self._model.decode_wav(
            y,
            int(sample_rate),
            phase_shift_decoding=self._phase_shift,
        )
        print_silentcipher_detection_summary(result=result, show_raw=bool(args.debug))
        if detect_log_path is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            row = detection_log_csv_row_silentcipher(
                audio_file,
                self.name,
                sample_rate=sample_rate,
                message_threshold=args.message_threshold,
                detection_threshold=args.detection_threshold,
                file_fraction_threshold=args.file_fraction_threshold,
                result=result,
                elapsed_ms=elapsed_ms,
            )
            with open(detect_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def attack_prepare(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        assert self._message is not None
        y = wav_batch_to_mono_numpy(wav_batched)
        sr = int(sample_rate)

        # Convert numpy to torch tensor on the same device as the model to avoid device mismatch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        y_tensor = torch.from_numpy(y).to(device)
        # calc_sdr=False: library sdr() mixes np.mean with torch tensors and crashes on recent numpy/torch.
        encoded, _sdr = self._model.encode_wav(y_tensor, sr, self._message, calc_sdr=False)
        # Move encoded back to CPU before converting to numpy
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()
        wav_wm = numpy_1d_to_batch_tensor(np.asarray(encoded, dtype=np.float32))

        if torch.cuda.is_available():
            wav_wm = wav_wm.cuda()
        return wav_wm, sr

    def attack_print_baseline(
        self,
        wav_wm: torch.Tensor,
        atk_sr: int,
        msg_t: float,
        det_t: float,
        frac_t: float,
    ) -> None:
        assert self._message is not None
        ok = silentcipher_message_recovered(
            self._model,
            self._message,
            wav_wm,
            atk_sr,
            phase_shift=self._phase_shift,
        )
        status = "message recovered" if ok else "message NOT recovered"
        print(
            f"  Baseline (watermarked, no attack): {status} "
            f"(SilentCipher exact 5-byte payload match)"
        )

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
        assert self._message is not None
        resisted = silentcipher_message_recovered(
            self._model,
            self._message,
            attacked_wm,
            atk_sr,
            phase_shift=self._phase_shift,
        )
        verdict = (
            "message still recovered (attack resisted)"
            if resisted
            else "message NOT recovered (attack succeeded)"
        )
        if attacked_wm.shape[-1] != wav_wm.shape[-1]:
            print(
                f"  Attack {attack_name!r}: {verdict}; "
                f"[length {wav_wm.shape[-1]} -> {attacked_wm.shape[-1]} samples]"
            )
        else:
            print(f"  Attack {attack_name!r}: {verdict}")
        return resisted
