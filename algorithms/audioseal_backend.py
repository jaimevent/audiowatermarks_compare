"""AudioSeal generator/detector integration."""

from __future__ import annotations

import csv
from typing import ClassVar

import torch
from audioseal import AudioSeal

from .base import WatermarkBackend


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
    """One row for detection log CSV; ``detected`` is ``X`` or ``-``."""
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


class AudiosealBackend(WatermarkBackend):
    name: ClassVar[str] = "audioseal"

    def setup(self, command: str) -> None:
        args = self.args
        self._generator = AudioSeal.load_generator(args.generator)
        self._generator.eval()
        self._detector = AudioSeal.load_detector(args.detector)
        self._detector.eval()
        if torch.cuda.is_available():
            self._generator = self._generator.cuda()
            self._detector = self._detector.cuda()

    @property
    def detector(self) -> torch.nn.Module:
        return self._detector

    def prepare_wav_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            return wav.cuda()
        return wav

    def embed_watermark(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        watermark = self._generator.get_watermark(wav_batched)
        watermarked = wav_batched + watermark
        return wav_batched, watermarked, sample_rate

    def finalize_watermarked_cuda(self, watermarked: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            return watermarked.cuda()
        return watermarked

    def compute_ber_nc_before_save(
        self,
        wav_batched: torch.Tensor,
        watermarked: torch.Tensor,
        sample_rate: int,
    ) -> tuple[float, float] | None:
        from bit_metrics import bit_error_rate, normalized_correlation

        _, bits_original = self._detector.detect_watermark(
            wav_batched,
            message_threshold=0.5,
            detection_threshold=0.5,
        )
        _, bits_watermarked = self._detector.detect_watermark(
            watermarked,
            message_threshold=0.5,
            detection_threshold=0.5,
        )
        ber_val = bit_error_rate(bits_original, bits_watermarked)
        nc_val = normalized_correlation(bits_original, bits_watermarked)
        return ber_val, nc_val

    def compute_ber_nc_after_save(self, wav_out_path: str) -> tuple[float, float] | None:
        return None

    def print_ber_line(self, ber_val: float) -> None:
        print(f"  BER: {ber_val:.4f}")

    def run_detect(
        self,
        wav_batched: torch.Tensor,
        sample_rate: int,
        audio_file: str,
        detect_log_path: str | None,
    ) -> None:
        args = self.args
        det_thresh = args.detection_threshold
        msg_t = args.message_threshold
        frac_t = args.file_fraction_threshold

        detect_frame_fraction, binary_message = self._detector.detect_watermark(
            wav_batched,
            message_threshold=msg_t,
            detection_threshold=det_thresh,
        )
        frame_logits, message_probs = self._detector(wav_batched)
        print_watermark_detection_summary(
            detect_frame_fraction=detect_frame_fraction,
            binary_message=binary_message,
            frame_logits=frame_logits,
            message_probs=message_probs,
            detection_threshold=det_thresh,
            file_fraction_threshold=frac_t,
            show_raw=bool(args.debug),
        )
        if detect_log_path is not None:
            row = detection_log_csv_row_audioseal(
                audio_file,
                self.name,
                sample_rate=sample_rate,
                message_threshold=msg_t,
                detection_threshold=det_thresh,
                file_fraction_threshold=frac_t,
                detect_frame_fraction=detect_frame_fraction,
                binary_message=binary_message,
                frame_logits=frame_logits,
                message_probs=message_probs,
            )
            with open(detect_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def attack_prepare(
        self, wav_batched: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        with torch.no_grad():
            wav_wm = wav_batched + self._generator.get_watermark(wav_batched)
        return wav_wm, sample_rate

    def attack_print_baseline(
        self,
        wav_wm: torch.Tensor,
        atk_sr: int,
        msg_t: float,
        det_t: float,
        frac_t: float,
    ) -> None:
        base_ok, base_frac = watermark_file_detected(
            self._detector,
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
        resisted, frac_wm = watermark_file_detected(
            self._detector,
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
        return resisted
