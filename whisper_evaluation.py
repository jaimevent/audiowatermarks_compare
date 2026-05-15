"""Whisper evaluation helpers for audio watermarking impact assessment.

This module uses OpenAI Whisper for evaluating WER/CER and RTF on raw and watermarked
audio datasets. It expects dataset roots to contain a matching split file path
provided via `--test-file`, with columns for audio paths and transcripts. Accepted
split-file extensions are `.csv`, `.tsv`, and `.txt`.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import julius
import librosa
import numpy as np
import soundfile as sf
import torch
import whisper

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")


@dataclass
class EvaluationMetrics:
    model_name: str
    dataset_root: str
    num_examples: int
    avg_wer: float
    avg_cer: float
    avg_rtf: float
    results_csv: str


def _resolve_test_file_path(dataset_root: str, test_file: str) -> str:
    candidate = test_file
    if not os.path.isabs(candidate):
        candidate = os.path.join(dataset_root, candidate)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Could not find split file {test_file} under dataset root {dataset_root}."
    )


def _sniff_csv_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
    if "\t" in sample and sample.count("\t") >= sample.count(","):
        return "\t"
    if "," in sample and sample.count(",") > sample.count("\t"):
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return dialect.delimiter
    except csv.Error:
        return ","


def _normalize_split_row(row: list[str]) -> list[str]:
    if len(row) == 1:
        text = row[0].strip()
        if not text:
            return []
        if "\t" in text:
            return [field.strip() for field in text.split("\t")]
        if " " in text:
            first, rest = text.split(None, 1)
            return [first.strip(), rest.strip()]
        return [text]
    return [field.strip() for field in row]


def _get_dataset_filenames(path: str) -> set[str]:
    """Build a set of basenames from the dataset's CSV for fast filtering."""
    filenames = set()
    delimiter = _sniff_csv_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            return filenames

        if len(header) == 1 and "\t" in header[0]:
            header = [column.strip() for column in header[0].split("\t")]
        header_lower = [column.strip().lower() for column in header]
        if "wav_filename" in header_lower and "transcript" in header_lower:
            wav_idx = header_lower.index("wav_filename")
        elif "path" in header_lower:
            wav_idx = header_lower.index("path")
        else:
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter)
            first_row = next(reader, None)
            if first_row is None:
                return filenames
            wav_idx = 0

        for row in reader:
            row = _normalize_split_row(row)
            if not row or row[0].startswith("#"):
                continue
            if len(row) <= wav_idx:
                continue
            audio_path = row[wav_idx]
            filenames.add(os.path.basename(audio_path))
    return filenames


def _filter_existing_files(dataset_root: str, filenames: set[str]) -> set[str]:
    """Filter a set of basenames to only those that exist in the dataset root.
    
    Searches for matching audio files with supported extensions (.wav, .flac, .mp3).
    """
    existing = set()
    for basename in filenames:
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            candidate = os.path.join(dataset_root, f"{basename}{ext}")
            if os.path.isfile(candidate):
                existing.add(basename)
                break
    return existing


def _iter_dataset_examples(path: str, whitelist_filenames: set[str] | None = None) -> Iterable[tuple[str, str]]:
    """Iterate dataset examples, optionally filtering by basename whitelist.
    
    Supports three file formats:
    1. TXT (space-separated): basename + transcript on each line
       Example: 103-1240-0000 CHAPTER ONE MISSUS RACHEL...
    2. TSV (tab-separated with header): must have 'path' and 'sentence' columns
       Example: client_id    path    sentence_id    sentence    ...
    3. CSV (comma-separated with header): 
       - With 'wav_filename' + 'transcript' columns, OR
       - With 'path' + ('sentence' or 'transcript') columns
    """
    delimiter = _sniff_csv_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            return

        if len(header) == 1 and "\t" in header[0]:
            header = [column.strip() for column in header[0].split("\t")]
        header_lower = [column.strip().lower() for column in header]
        if "wav_filename" in header_lower and "transcript" in header_lower:
            wav_idx = header_lower.index("wav_filename")
            transcript_idx = header_lower.index("transcript")
            for row in reader:
                row = _normalize_split_row(row)
                if not row or row[0].startswith("#"):
                    continue
                if len(row) <= max(wav_idx, transcript_idx):
                    continue
                audio_path = row[wav_idx]
                if whitelist_filenames and os.path.basename(audio_path) not in whitelist_filenames:
                    continue
                yield audio_path, row[transcript_idx]
            return
        elif "path" in header_lower:
            wav_idx = header_lower.index("path")
            if "sentence" in header_lower:
                transcript_idx = header_lower.index("sentence")
            elif "transcript" in header_lower:
                transcript_idx = header_lower.index("transcript")
            else:
                raise ValueError(
                    f"Dataset {path} has a 'path' column but no 'sentence' or 'transcript' column."
                )
            for row in reader:
                row = _normalize_split_row(row)
                if not row or row[0].startswith("#"):
                    continue
                if len(row) <= max(wav_idx, transcript_idx):
                    continue
                audio_path = row[wav_idx]
                if whitelist_filenames and os.path.basename(audio_path) not in whitelist_filenames:
                    continue
                yield audio_path, row[transcript_idx]
            return
        else:
            # Headerless file: TXT format (space-separated basename + transcript).
            # Example: "103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE..."
            # After _normalize_split_row, this becomes ['103-1240-0000', 'CHAPTER ONE...']
            csv_file = Path(path)
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter)
            first_row = next(reader, None)
            if first_row is None:
                return
            if len(first_row) >= 3 and first_row[1].strip().isdigit():
                wav_idx, transcript_idx = 0, 2
            else:
                wav_idx, transcript_idx = 0, 1
            for row in reader:
                row = _normalize_split_row(row)
                if not row or row[0].startswith("#"):
                    continue
                if len(row) <= max(wav_idx, transcript_idx):
                    continue
                audio_path = row[wav_idx]
                if whitelist_filenames and os.path.basename(audio_path) not in whitelist_filenames:
                    continue
                yield audio_path, row[transcript_idx]
            return


def _find_audio_file_by_basename(dataset_root: str, audio_path: str) -> str | None:
    directory = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    if os.path.splitext(basename)[1]:
        return None

    search_root = os.path.join(dataset_root, directory) if directory else dataset_root
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = os.path.join(search_root, f"{basename}{ext}")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def _resolve_audio_path(dataset_root: str, audio_path: str) -> str:
    if os.path.isabs(audio_path):
        if os.path.isfile(audio_path):
            return audio_path
        fallback = _find_audio_file_by_basename(os.path.dirname(audio_path), audio_path)
        return fallback if fallback is not None else audio_path

    candidate = os.path.abspath(os.path.join(dataset_root, audio_path))
    if os.path.isfile(candidate):
        return candidate
    if os.path.splitext(audio_path)[1]:
        return candidate
    fallback = _find_audio_file_by_basename(dataset_root, audio_path)
    return fallback if fallback is not None else candidate


def _whisper_language_for_transcribe(language: str | None) -> str | None:
    """Map user/API language to Whisper's ``language`` (``None`` = auto-detect)."""
    if language is None:
        return None
    code = language.strip().lower()
    if code in ("auto", "none", ""):
        return None
    return code


def _normalize_transcript(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = _normalize_transcript(reference).split()
    hyp_words = _normalize_transcript(hypothesis).split()
    if not ref_words:
        return float("inf") if hyp_words else 0.0
    dist = _compute_edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(_normalize_transcript(reference).replace(" ", ""))
    hyp_chars = list(_normalize_transcript(hypothesis).replace(" ", ""))
    if not ref_chars:
        return float("inf") if hyp_chars else 0.0
    dist = _compute_edit_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def _compute_edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(1, len(ref_tokens) + 1):
        dp[i][0] = i
    for j in range(1, len(hyp_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(hyp_tokens) + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _resample_audio_to_rate(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio
    audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    resampled = julius.resample_frac(audio_tensor, original_sr, target_sr).squeeze(0).numpy()
    return resampled.astype(audio.dtype)


def _load_audio_for_whisper(file_path: str, sample_rate: int) -> np.ndarray:
    """Load mono float32 audio; MP3 via librosa (soundfile/libsndfile often lacks MP3)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp3":
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
    else:
        audio, sr = sf.read(file_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = _resample_audio_to_rate(audio, sr, sample_rate)
    return audio


def evaluate_with_whisper(
    dataset_root: str,
    results_csv: str,
    model_size: str = "base",
    sample_rate: int = 16000,
    *,
    language: str | None = "it",
    test_csv_path: str | None = None,
    whitelist_filenames: set[str] | None = None,
) -> EvaluationMetrics:
    """Evaluate WER/CER and RTF using Whisper on the test set.

    Real-Time Factor (RTF) is wall-clock transcribe time divided by audio duration
    (below 1.0 means faster than real time).

    ``language`` is an ISO 639-1 code (e.g. ``it``, ``en``). Use ``None`` or the string
    ``auto`` for automatic language detection.

    If ``whitelist_filenames`` is provided, only files with matching basenames are evaluated.
    """
    model = whisper.load_model(model_size)
    if test_csv_path is None:
        raise ValueError("test_csv_path must be provided for Whisper evaluation.")
    test_csv = test_csv_path
    whisper_lang = _whisper_language_for_transcribe(language)

    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    total_wer = 0.0
    total_cer = 0.0
    total_rtf = 0.0
    count = 0
    rtf_count = 0

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["audio_file", "reference", "hypothesis", "wer", "cer", "rtf"]
        )
        for wav_path, transcript in _iter_dataset_examples(test_csv, whitelist_filenames):
            absolute_wav = _resolve_audio_path(dataset_root, wav_path)
            if not os.path.isfile(absolute_wav):
                #raise FileNotFoundError(f"Audio file not found: {absolute_wav}")
                #print(f"Warning: Audio file not found, skipping: {absolute_wav}")
                continue
            audio = _load_audio_for_whisper(absolute_wav, sample_rate)
            duration_sec = len(audio) / float(sample_rate)
            t0 = time.perf_counter()
            result = model.transcribe(
                audio,
                language=whisper_lang,
                fp16=torch.cuda.is_available(),
            )
            elapsed_sec = time.perf_counter() - t0
            if duration_sec > 0:
                rtf = elapsed_sec / duration_sec
                total_rtf += rtf
                rtf_count += 1
            else:
                rtf = float("nan")
            hypothesis = result["text"]
            wer = compute_wer(transcript, hypothesis)
            cer = compute_cer(transcript, hypothesis)
            total_wer += wer
            total_cer += cer
            count += 1

            print(f"Completed transcribing {wav_path} in {elapsed_sec:.2f} seconds")
            print(f"RTF: {rtf:.6f}")
            print(f"WER: {wer:.6f}")
            print(f"CER: {cer:.6f}")
            print(f"Transcript: {hypothesis}")
            print(f"Reference: {transcript}")
            print("-" * 100)

            writer.writerow(
                [
                    absolute_wav,
                    transcript,
                    hypothesis,
                    f"{wer:.6f}",
                    f"{cer:.6f}",
                    f"{rtf:.6f}" if not np.isnan(rtf) else "nan",
                ]
            )

    avg_wer = total_wer / count if count else float("nan")
    avg_cer = total_cer / count if count else float("nan")
    avg_rtf = total_rtf / rtf_count if rtf_count else float("nan")
    return EvaluationMetrics(
        model_name=f"whisper-{model_size}",
        dataset_root=dataset_root,
        num_examples=count,
        avg_wer=avg_wer,
        avg_cer=avg_cer,
        avg_rtf=avg_rtf,
        results_csv=os.path.abspath(results_csv),
    )


def evaluate_datasets(
    raw_dataset_root: str,
    watermarked_dataset_root: str,
    test_file: str,
    output_root: str,
    model_size: str = "base",
    sample_rate: int = 16000,
    *,
    language: str | None = "it",
) -> list[EvaluationMetrics]:
    """Evaluate WER/CER/RTF on raw and watermarked datasets using Whisper.
    
    Watermarked directory is used as reference: raw files are only evaluated if
    they have a matching basename in the watermarked directory.
    """
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    results: list[EvaluationMetrics] = []

    # Evaluate watermarked dataset first to establish the reference set
    watermarked_csv = _resolve_test_file_path(
        watermarked_dataset_root,
        test_file,
    )
    watermarked_filenames = _get_dataset_filenames(watermarked_csv)
    print(f"Found {len(watermarked_filenames)} entries in watermarked test file")
    watermarked_filenames = _filter_existing_files(watermarked_dataset_root, watermarked_filenames)
    print(f"Found {len(watermarked_filenames)} matching audio files in watermarked dataset folder")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    watermarked_csv_out = os.path.join(output_root, f"{stamp}_watermarked_whisper_evaluation.csv")
    print("Evaluating watermarked dataset...")
    metrics = evaluate_with_whisper(
        dataset_root=watermarked_dataset_root,
        results_csv=watermarked_csv_out,
        model_size=model_size,
        sample_rate=sample_rate,
        language=language,
        test_csv_path=watermarked_csv,
    )
    results.append(metrics)

    # Evaluate raw dataset, using watermarked filenames as reference filter
    raw_csv = _resolve_test_file_path(
        raw_dataset_root,
        test_file,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_out = os.path.join(output_root, f"{stamp}_raw_whisper_evaluation.csv")
    print(f"Evaluating raw dataset (filtered to {len(watermarked_filenames)} matching files)...")
    metrics = evaluate_with_whisper(
        dataset_root=raw_dataset_root,
        results_csv=raw_csv_out,
        model_size=model_size,
        sample_rate=sample_rate,
        language=language,
        test_csv_path=raw_csv,
        whitelist_filenames=watermarked_filenames,
    )
    results.append(metrics)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = os.path.join(output_root, f"{stamp}_whisper_comparison_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_label",
            "dataset_root",
            "results_csv",
            "num_examples",
            "avg_wer",
            "avg_cer",
            "avg_rtf",
        ])
        for metrics in results:
            writer.writerow(
                [
                    metrics.model_name,
                    metrics.dataset_root,
                    metrics.results_csv,
                    metrics.num_examples,
                    f"{metrics.avg_wer:.6f}",
                    f"{metrics.avg_cer:.6f}",
                    f"{metrics.avg_rtf:.6f}",
                ]
            )

    return results