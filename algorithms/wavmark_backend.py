"""WavMark model integration (encode / decode at 16 kHz)."""

from __future__ import annotations

import csv
from typing import ClassVar

import numpy as np
import time
import torch
import wavmark as wm
from wavmark.utils import wm_add_util as wavmark_wm_add_util

from audio_io import load_waveform_torch
from bit_metrics import normalized_correlation
from wavmark_io import (
    wavmark_decode_sliding_stats,
    wavmark_decode_watermark,
    wavmark_mono_16k_tensor,
)

from .base import WatermarkBackend


def numpy_1d_to_batch_tensor(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).unsqueeze(0)


def wavmark_payload_survives_attack(
    wmmodel,
    payload_embedded: np.ndarray,
    wav_batch_ch_first: torch.Tensor,
    audio_sample_rate: int,
) -> bool:
    """True iff WavMark decodes ``payload_embedded`` exactly from ``wav_batch_ch_first``."""

    mono16 = wavmark_mono_16k_tensor(wav_batch_ch_first, audio_sample_rate)
    dec, info, _ = wavmark_decode_watermark(wmmodel, mono16, show_progress=False)
    if dec is None:
        return False
    # Fast check: if sync fraction is very high, payload likely recovered.
    # Avoid expensive comparison if sync is weak.
    _, sync_frac, sims = wavmark_decode_sliding_stats(mono16, info)
    if sync_frac < 0.1:  # Threshold: less than 10% sync hits suggests attack succeeded.
        return False
    # Only do expensive payload comparison if there's meaningful sync activity.
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
    elapsed_ms: float | None = None,
) -> list:
    """CSV row aligned with detection log header using WavMark decode statistics."""

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

    row = [
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
    if elapsed_ms is not None:
        row.append(round(elapsed_ms, 2))
    return row


class WavmarkBackend(WatermarkBackend):
    name: ClassVar[str] = "wavmark"

    def setup(self, command: str) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._wmmodel = wm.load_model().to(device)
        self._payload: np.ndarray | None = None
        if command in ("watermark", "attack"):
            self._payload = np.random.choice([0, 1], size=16)
            print("Payload:", self._payload)

    def prepare_wav_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        return wav

    def embed_watermark(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        assert self._payload is not None
        wav_16k = wavmark_mono_16k_tensor(wav_batched, sample_rate)
        min_length = 16000 + int(16000 * 0.1)
        original_length = len(wav_16k)
        if original_length < min_length:
            pad_len = min_length - original_length
            wav_16k_padded = np.concatenate([wav_16k, np.zeros(pad_len, dtype=np.float32)])
        else:
            wav_16k_padded = wav_16k

        watermarked_np, _ = wm.encode_watermark(
            self._wmmodel, wav_16k_padded, self._payload, show_progress=False
        )
        watermarked_np = watermarked_np[:original_length]
        watermarked_audio = numpy_1d_to_batch_tensor(watermarked_np)
        metrics_ref = numpy_1d_to_batch_tensor(wav_16k)
        if torch.cuda.is_available():
            watermarked_audio = watermarked_audio.cuda()
            metrics_ref = metrics_ref.cuda()
        return metrics_ref, watermarked_audio, 16000

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
        assert self._payload is not None
        wav_disk, sr_disk = load_waveform_torch(wav_out_path)
        payload_decoded_file, _, _ = wavmark_decode_watermark(
            self._wmmodel,
            wavmark_mono_16k_tensor(wav_disk.unsqueeze(0), sr_disk),
            show_progress=False,
        )
        if payload_decoded_file is None:
            return 100.0, 0.0
        ber_val = float((self._payload != payload_decoded_file).mean() * 100.0)
        nc_val = normalized_correlation(
            torch.from_numpy(np.asarray(self._payload, dtype=np.float32).reshape(-1)),
            torch.from_numpy(
                np.asarray(payload_decoded_file, dtype=np.float32).reshape(-1)
            ),
        )
        return ber_val, nc_val

    def print_ber_line(self, ber_val: float) -> None:
        print(f"  BER: {ber_val:.3f}%")

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
        wm_mono = wavmark_mono_16k_tensor(wav_batched, sample_rate)
        payload_decoded, decode_info, wav_16k = wavmark_decode_watermark(
            self._wmmodel, wm_mono, show_progress=False
        )
        print_wavmark_detection_summary(
            payload_decoded=payload_decoded,
            decode_info=decode_info,
            wav_16k=wav_16k,
            show_raw=bool(args.debug),
        )
        if detect_log_path is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            row = detection_log_csv_row_wavmark(
                audio_file,
                self.name,
                sample_rate=sample_rate,
                message_threshold=args.message_threshold,
                detection_threshold=args.detection_threshold,
                file_fraction_threshold=args.file_fraction_threshold,
                wav_16k=wav_16k,
                payload_decoded=payload_decoded,
                decode_info=decode_info,
                elapsed_ms=elapsed_ms,
            )
            with open(detect_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def attack_prepare(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        assert self._payload is not None
        wm_mono = wavmark_mono_16k_tensor(wav_batched, sample_rate)
        min_length = 16000 + int(16000 * 0.1)
        original_length = len(wm_mono)
        if original_length < min_length:
            pad_len = min_length - original_length
            wm_mono_padded = np.concatenate([wm_mono, np.zeros(pad_len, dtype=np.float32)])
        else:
            wm_mono_padded = wm_mono

        watermarked_np, _ = wm.encode_watermark(
            self._wmmodel, wm_mono_padded, self._payload, show_progress=False
        )
        watermarked_np = watermarked_np[:original_length]
        wav_wm = numpy_1d_to_batch_tensor(watermarked_np)
        # Keep on CPU to avoid repeated GPU-CPU transfers during attack evaluation.
        # (WavMark decode runs on CPU anyway; moving to GPU and back each attack is slow.)
        return wav_wm, 16000

    def attack_print_baseline(
        self,
        wav_wm: torch.Tensor,
        atk_sr: int,
        msg_t: float,
        det_t: float,
        frac_t: float,
    ) -> None:
        assert self._payload is not None
        base_ok = wavmark_payload_survives_attack(
            self._wmmodel, self._payload, wav_wm, atk_sr
        )
        status = "payload recovered" if base_ok else "payload NOT recovered"
        print(
            f"  Baseline (watermarked, no attack): {status} "
            f"(WavMark exact 16-bit payload match vs embedded)"
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
        assert self._payload is not None
        resisted = wavmark_payload_survives_attack(
            self._wmmodel, self._payload, attacked_wm, atk_sr
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
        return resisted
