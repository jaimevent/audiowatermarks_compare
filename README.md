# Audiowatermarks Compare

Python utility to benchmark audio watermarking algorithms on folders of audio files. Supports embedding watermarks, detection, robustness testing against attacks, and evaluating impact on automatic speech recognition using **OpenAI Whisper** (WER, CER, and real-time factor).

## Supported algorithms

| ID | Method | Backend module |
|----|--------|----------------|
| `audioseal` | [Meta AudioSeal](https://github.com/facebookresearch/audioseal) | `algorithms/audioseal_backend.py` |
| `silentcipher` | [Sony SilentCipher](https://github.com/SesameAILabs/silentcipher) (SesameAILabs fork for PyTorch 2.x) | `algorithms/silentcipher_backend.py` |
| `wavmark` | [WavMark](https://github.com/wavmark/wavmark) | `algorithms/wavmark_backend.py` |
| `dsss` | Direct-sequence spread-spectrum (classical PN embedding) | `algorithms/dsss_backend.py` |

New algorithms: implement `WatermarkBackend` in `algorithms/`, register in `algorithms/__init__.py` (`ALGORITHM_REGISTRY`).

## Requirements

- **Python** 3.10 or newer.
- **PyTorch**, **torchaudio**, **torchcodec**, **audioseal**, **soundfile**, **matplotlib**, **librosa**, **julius**, **pesq**, and other dependencies (see `requirements.txt`).

Install PyTorch from the [official install page](https://pytorch.org/get-started/locally/) if you need a specific CUDA build; then install the rest:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

On Windows, if you use the CPU wheel index for `torchcodec`, match it to your PyTorch channel:

```bash
pip install torchcodec --index-url=https://download.pytorch.org/whl/cpu
```

**MP3 output** (watermark with `--output-format match-input`) needs **FFmpeg** in `PATH` or a working TorchAudio MP3 backend.

### OpenAI Whisper (`evaluate` command)

```bash
pip install openai-whisper
```

Whisper downloads model weights on first use.

## Invoking the program

```bash
python main.py COMMAND [COMMAND_OPTIONS]
```

Backend-related global options (`--generator`, `--detector`, `--silentcipher-*`) are defined on the root parser and apply to **`watermark`**, **`detect`**, and **`attack`**. They are ignored by **`evaluate`**.

| Command | Role |
|---------|------|
| `watermark` | Embed watermarks; SNR, PESQ, BER, NC; optional waveform plots and metrics CSV. |
| `detect` | Run the chosen backend’s detector; append rows to a detection log CSV. |
| `attack` | Embed, apply attacks from `attacks.AudioEffects`, test recovery; attack metrics CSV and summary plots. |
| `evaluate` | Whisper WER/CER/RTF on raw vs watermarked dataset roots. |

Input folder: top-level **`.wav`**, **`.flac`**, and **`.mp3`** only (non-recursive).

---

## Command: `watermark`

**Required:** `-i DIR` / `--input DIR`

| Option | Default | Description |
|--------|---------|-------------|
| `-a`, `--algorithm` | `audioseal` | `audioseal`, `dsss`, `silentcipher`, `wavmark` |
| `--all-algorithms` | off | Run every registered algorithm in one invocation |
| `--max-audios N` | `1000000` | Cap number of files processed |
| `-o`, `--output-plot DIR` | `plots` (cwd) | PNG comparison plots (`<stem>_<algorithm>_original_vs_watermarked.png`) |
| `--no-plots` | off | Skip plots |
| `--plot-dpi N` | `150` | Plot resolution |
| `--output-watermarked DIR` | `<input>/<algorithm>_watermarked_wav` | Watermarked audio output |
| `--output-metrics FILE` | `metrics/<timestamp>_…_watermark_metrics.csv` | SNR, PESQ, BER, NC per file |
| `--output-format {wav,match-input}` | `wav` | Always `.wav`, or match source extension |

**Per-run metrics (console + CSV):**

- **SNR (dB)** — original vs additive watermark (`watermarked − original`).
- **PESQ** — perceptual quality (resampled to 16 kHz when needed).
- **BER / NC** — backend-specific; see [Metrics conventions](#metrics-conventions).

**Global / AudioSeal** (root parser): `--generator`, `--detector`, `--debug`

**SilentCipher** (root): `--silentcipher-model {44.1k,16k}`, `--silentcipher-phase-shift`

**DSSS** (`--algorithm dsss` only):

| Option | Default | Description |
|--------|---------|-------------|
| `--dsss-message TEXT` | `unimilano` | UTF-8 payload embedded as bits |
| `--dsss-frame-length N` | `4096` | Samples per spread-spectrum bit |
| `--dsss-alpha A` | `0.01` | Embedding strength |
| `--dsss-seed N` | `42` | PRNG seed for PN sequences (per-frame `seed + frame_index`) |

Short clips are **zero-padded** at the end so the full payload fits one repetition.

**Example — DSSS:**

```bash
python main.py watermark -i path/to/audio -a dsss ^
  --dsss-message "unimilano" --dsss-alpha 0.02 --dsss-frame-length 8192 ^
  --output-plot plots --output-watermarked out/dsss_wm
```

---

## Command: `detect`

**Required:** `-i DIR`

Same `--algorithm` / `--all-algorithms` / `--max-audios` as watermark.

| Option | Default | Description |
|--------|---------|-------------|
| `--detection-threshold` | `0.5` | Frame-level P(watermark) (AudioSeal-style backends) |
| `--message-threshold` | `0.5` | Message bit threshold |
| `--file-fraction-threshold` | `0.5` | File detected if enough frames exceed detection threshold |
| `--output-detect-log FILE` | timestamped under `metrics/` | Detection log CSV |

**DSSS:** `--dsss-message` (must match embedding). Frame length, alpha, and seed use backend defaults (`4096`, `0.01`, `42`) unless you align them with the values used at embed time — use the same defaults on `watermark` and `detect`, or pass matching flags on `watermark`/`attack` where available.

```bash
python main.py detect -i path/to/dsss_watermarked_wav -a dsss --dsss-message "unimilano"
```

---

## Command: `attack`

**Required:** `-i DIR`

Detection thresholds and algorithm options as for `detect`. Additional:

| Option | Default | Description |
|--------|---------|-------------|
| `--attack-seed N` | `42` | RNG seed before each stochastic attack |
| `--save-attacked DIR` | (none) | Save attacked WAVs under `<DIR>/<stem>/` |
| `--output-attack-metrics FILE` | timestamped under `metrics/` | Per-file attack resistance matrix |
| `-o`, `--output-attack-plot DIR` | `plots/attack_summary` | Heatmap + bar chart per algorithm |
| `--no-attack-plots` | off | Skip summary PNGs |
| `--attack-plot-dpi N` | `150` | Plot resolution |

**DSSS** (same as watermark for embed/extract during the attack pipeline):

| Option | Default |
|--------|---------|
| `--dsss-message` | `unimilano` |
| `--dsss-frame-length` | `4096` |
| `--dsss-alpha` | `0.01` |
| `--dsss-seed` | `42` |

Use the **same** DSSS settings as when you embedded watermarks you want to compare against.

```bash
python main.py attack -i path/to/audio -a dsss ^
  --dsss-message "unimilano" --dsss-alpha 0.01 --save-attacked attacked_out
```

---

## Command: `evaluate` (Whisper)

**Required**

| Option | Description |
|--------|-------------|
| `--raw-dataset DIR` | Raw dataset with `test.csv` or `test.tsv` |
| `--watermarked-dataset DIR` | Watermarked dataset (same layout) |

**Optional**

| Option | Default | Description |
|--------|---------|-------------|
| `--output-root DIR` | `whisper_results` | Output folder |
| `--model-size` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `--sample-rate` | `16000` | Resample before transcription |
| `--language CODE` | `it` | ISO 639-1 (`en`, `it`, …) or `auto` |
| `--max-audios N` | `1000000` | Maximum number of audio files to evaluate per dataset |

**Outputs** (under `--output-root`):

| File | Contents |
|------|----------|
| `raw_whisper_evaluation.csv` | Per utterance: `audio_file`, `reference`, `hypothesis`, `wer`, `cer`, `rtf` |
| `watermarked_whisper_evaluation.csv` | Same for watermarked set |
| `whisper_comparison_summary.csv` | `avg_wer`, `avg_cer`, `avg_rtf` per dataset |

**WER/CER** are fractions in **0–1** (not percent). **RTF** = transcribe time / audio duration; **&lt; 1** is faster than real time.

Audio in split files: **`.wav`**, **`.flac`**, **`.mp3`** (MP3 via librosa).

```bash
python main.py evaluate ^
  --raw-dataset path/to/raw ^
  --watermarked-dataset path/to/wm ^
  --output-root whisper_results ^
  --model-size base --language it
```

---

## Quick examples

```bash
# Default (AudioSeal)
python main.py watermark -i path/to/audio_folder

# All algorithms
python main.py watermark -i path/to/audio_folder --all-algorithms

# Detection / attacks
python main.py detect -i path/to/audio_folder -a wavmark
python main.py attack -i path/to/audio_folder -a dsss

# Whisper
python main.py evaluate --raw-dataset path/to/raw --watermarked-dataset path/to/wm
```

---

## What the script does

### Watermark embedding

1. Load audio (WAV/FLAC/MP3 via `audio_io`).
2. Embed with the selected backend.
3. Print SNR and PESQ; BER/NC when the backend provides them.
4. Save watermarked audio (and optional plots / metrics CSV).

### Detection

Runs the backend decoder and appends a row to the detection log CSV (`X` = detected, `-` = not).

### Attacks

Embeds a watermark, applies each attack in `attacks.AudioEffects`, tests whether the payload still decodes, writes CSV and optional robustness heatmap/bar charts.

### Whisper evaluation

Transcribes test splits, computes WER/CER/RTF, writes detailed and summary CSVs.

---

## DSSS algorithm (summary)

Classical **direct-sequence spread spectrum**: each payload bit modulates a pseudo-noise (±1) sequence over `frame_length` samples; detection correlates each frame with the same PN (`seed + frame_index`).

- Implementation: `algorithms/dsss_backend.py`
- Mono mixdown for multi-channel inputs
- BER/NC measured on **payload bits** after round-trip through the saved file (`watermark` command)
- **Attack resistance**: full UTF-8 message match after each attack

---

## Metrics conventions

| Backend | BER in CSV / logs | Notes |
|---------|-------------------|--------|
| `audioseal` | 0–1 fraction | Detector bits: original vs watermarked |
| `dsss` | 0–1 fraction | Embedded vs extracted payload bits |
| `wavmark`, `silentcipher` | 0–100 percent | Payload bit errors |

**NC** — normalized correlation between reference and estimated bit vectors (see `bit_metrics.py`).

---

## Behaviour notes

### Audio loading

- **WAV**, **FLAC**, **MP3** supported (`soundfile`, `librosa` for MP3).
- Tensors are `(batch, channels, samples)`; backends may downmix to mono.

### Length limits

- Files longer than **60 s at 16 kHz equivalent** (`16000 × 60` samples) are skipped to limit memory use.

### PyTorch compilation

- `TORCHDYNAMO_DISABLE=1` is set in `main.py` to avoid Inductor/MSVC issues on Windows.

### Metrics output directory

Timestamped CSVs are written under `metrics/` (see `metrics_csv.py`).

---

## Project layout

```
main.py                 CLI entry point
audio_io.py             Load/save WAV, FLAC, MP3
attacks.py              Attack suite for `attack`
bit_metrics.py          BER, NC helpers
metrics_csv.py          CSV headers and paths
watermark_plots.py      Waveform plots and attack summaries
whisper_evaluation.py   Whisper WER/CER/RTF
algorithms/             Pluggable watermark backends
```

---

## References

- AudioSeal: [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)
- SilentCipher: [SesameAILabs/silentcipher](https://github.com/SesameAILabs/silentcipher)
- WavMark: [wavmark/wavmark](https://github.com/wavmark/wavmark)
- Whisper: [openai/whisper](https://github.com/openai/whisper)

## License

Scripts for research use; third-party algorithms follow their upstream licenses.
