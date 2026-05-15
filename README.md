# Audiowatermarks Compare

Python utility to benchmark audio watermarking algorithms on folders of audio files. Supports embedding watermarks, detection, robustness testing against attacks, and evaluating impact on automatic speech recognition using **OpenAI Whisper** (WER, CER, and real-time factor).

Currently supported algorithms:

- **Meta AudioSeal** (facebookresearch/audioseal)
- **Sony SilentCipher** (SesameAILabs fork for PyTorch 2.x compatibility)
- **WavMark** (watermarking via adversarial examples)

## Requirements

- **Python** 3.10 or newer.
- **PyTorch**, **torchaudio**, **torchcodec**, **audioseal**, **soundfile**, **matplotlib**, and other dependencies (see `requirements.txt`).

Install PyTorch from the [official install page](https://pytorch.org/get-started/locally/) if you need a specific CUDA build; then install the rest:

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

On Windows, if you use the CPU wheel index for `torchcodec`, match it to your PyTorch channel, for example:

```bash
pip install torchcodec --index-url=https://download.pytorch.org/whl/cpu
```

### OpenAI Whisper (for `evaluate` command)

The `evaluate` command uses OpenAI Whisper for WER, CER, and **RTF** (real-time factor) on raw and watermarked datasets.

```bash
pip install openai-whisper
```

Whisper downloads model weights on first use.

## Invoking the program

General form:

```bash
python main.py [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

**Global options** (parsed before the subcommand) configure backends used by **`watermark`**, **`detect`**, and **`attack`**. They are accepted for **`evaluate`** as well but have no effect there (Whisper evaluation does not use the watermark backends).

| Option | Description |
|--------|-------------|
| `--generator NAME` | AudioSeal generator card name (default: `audioseal_wm_16bits`). |
| `--detector NAME` | AudioSeal detector card name (default: `audioseal_detector_16bits`). |
| `--debug` | Show debug information (default: enabled). |
| `--max-audios` | Maximum number of audio files to process (default: 1.000.000). |
| `--silentcipher-model {44.1k,16k}` | SilentCipher checkpoint family (default: `44.1k`). Ignored unless `--algorithm silentcipher`. |
| `--silentcipher-phase-shift` | Use phase-shift decoding for SilentCipher on watermark/detect (`decode_wav(..., phase_shift_decoding=True)`); slower, more robust to crops. Ignored unless algorithm is SilentCipher. |

### Commands

| Command | Role |
|---------|------|
| `watermark` | Embed watermarks and record quality metrics (SNR, PESQ, BER, NC, plots). |
| `detect` | Run watermark detection on audio files; log results to CSV. |
| `attack` | Embed, apply attacks, detect; optionally save attacked audio and robustness plots. |
| `evaluate` | Run Whisper on **raw** and **watermarked** dataset roots; write per-utterance CSVs and a comparison summary. |

---

## Command: `evaluate` (Whisper)

**Required**

| Option | Description |
|--------|-------------|
| `--raw-dataset DIR` | Directory containing the **raw** dataset (must include a `test.csv` or `test.tsv` split). |
| `--watermarked-dataset DIR` | Same layout for **watermarked** audio and references. |
| `--test-file FILE` | | Split file path shared by both datasets. If relative, this path is resolved under each dataset root. Each row may contain a relative audio path or a basename without extension. Accepted extensions: `.csv`, `.tsv`, `.txt`. |

**Optional**

| Option | Default | Description |
|--------|---------|-------------|
| `--output-root DIR` | `whisper_results` | Folder for all Whisper outputs (CSVs below). |
| `--model-size SIZE` | `base` | Whisper size: `tiny`, `base`, `small`, `medium`, `large`. |
| `--sample-rate SR` | `16000` | Resample loaded audio to this rate before transcription. |
| `--language CODE` | `it` | ISO 639-1 language passed to Whisper (e.g. `it`, `en`). Use `auto` (or `none`) for automatic language detection. |

**Example**

```bash
python main.py evaluate ^
  --raw-dataset path/to/raw_dataset ^
  --watermarked-dataset path/to/wm_dataset ^
  --output-root whisper_results ^
  --model-size base ^
  --sample-rate 16000 ^
  --language it
```

(On Unix, replace `^` with `\` or put arguments on one line.)

**Outputs** (under `--output-root`)

| File | Contents |
|------|----------|
| `raw_whisper_evaluation.csv` | One row per test utterance: `audio_file`, `reference`, `hypothesis`, `wer`, `cer`, **`rtf`**. |
| `watermarked_whisper_evaluation.csv` | Same columns for the watermarked set. |
| `whisper_comparison_summary.csv` | One row per dataset: `model_label`, `dataset_root`, `results_csv`, `num_examples`, `avg_wer`, `avg_cer`, **`avg_rtf`**. |

**RTF (real-time factor)** is wall-clock Whisper transcribe time divided by audio duration: **RTF &lt; 1** means faster than real time, **&gt; 1** means slower. Per-utterance RTF is averaged into **`avg_rtf`** in the summary.

---

## Command: `watermark`

**Required**

| Option | Description |
|--------|-------------|
| `-i DIR`, `--input DIR` | Folder with `.wav` / `.flac` / `.mp3` (top level only; no subfolders). |

**Algorithm**

| Option | Default | Description |
|--------|---------|-------------|
| `-a ALGORITHM`, `--algorithm` | `audioseal` | One of: `audioseal`, `silentcipher`, `wavmark`. |
| `--all-algorithms` | off | Run every registered algorithm in one invocation. |

**Paths and outputs**

| Option | Default | Description |
|--------|---------|-------------|
| `-o DIR`, `--output-plot` | `plots` (cwd) | Directory for waveform/spectrogram comparison PNGs. |
| `--no-plots` | off | Skip plots. |
| `--plot-dpi N` | `150` | PNG resolution. |
| `--output-watermarked DIR` | `<input>/<algorithm>_watermarked_wav` | Where to write watermarked files. |
| `--output-metrics FILE` | timestamped under `metrics/` | Basename for watermark metrics CSV (see `metrics_csv` helpers). |
| `--output-format {wav,match-input}` | `wav` | Output always `.wav`, or match each source extension (`.wav`/`.flac`/`.mp3`; MP3 may need FFmpeg). |

---

## Command: `detect`

**Required:** `-i` / `--input` as for `watermark`.

**Algorithm:** `--algorithm`, `--all-algorithms` (same as above).

**Detection thresholds**

| Option | Default | Description |
|--------|---------|-------------|
| `--detection-threshold P` | `0.5` | Frame-level P(watermark) threshold. |
| `--message-threshold P` | `0.5` | Message bit threshold for detection. |
| `--file-fraction-threshold FRAC` | `0.5` | Fraction of frames above `--detection-threshold` required to count as detected. |

**Logging**

| Option | Default | Description |
|--------|---------|-------------|
| `--output-detect-log FILE` | timestamped under `metrics/` | Basename for detection log CSV. |

---

## Command: `attack`

**Required:** `-i` / `--input` as above.

**Algorithm:** `--algorithm`, `--all-algorithms`.

**Detection:** same threshold options as `detect`.

**Attack-specific**

| Option | Default | Description |
|--------|---------|-------------|
| `--attack-seed N` | `42` | RNG seed for stochastic attacks. |
| `--save-attacked DIR` | (none) | Save attacked watermarked WAVs (per-file subfolders). |
| `--output-attack-metrics FILE` | timestamped under `metrics/` | Attack metrics CSV. |
| `-o DIR`, `--output-attack-plot` | `plots/attack_summary` (cwd) | Heatmap and bar chart PNGs. |
| `--no-attack-plots` | off | Skip attack summary plots. |
| `--attack-plot-dpi N` | `150` | Resolution of attack summary figures. |

---

## Quick examples

Embed watermarks (default AudioSeal):

```bash
python main.py watermark -i path/to/audio_folder
```

Detect:

```bash
python main.py detect -i path/to/audio_folder
```

Robustness pipeline:

```bash
python main.py attack -i path/to/audio_folder
```

Whisper evaluation (raw vs watermarked datasets):

```bash
python main.py evaluate \
  --raw-dataset /path/to/raw \
  --watermarked-dataset /path/to/watermarked \
  --test-file test.csv
```

Run all algorithms on watermark:

```bash
python main.py watermark -i path/to/audio_folder --all-algorithms
```

---

## What the script does

### Watermark embedding

1. Load audio (WAV directly; FLAC/MP3 via loaders as implemented).
2. Embed watermark.
3. Print SNR (dB) and PESQ.
4. Save watermarked audio.
5. BER/NC where supported.
6. Optional comparison plots.

### Detection

Runs the chosen backend’s detector and appends rows to the detection log CSV.

### Attacks

Embeds watermarks, applies the configured attack suite, runs detection, and optionally exports robustness plots.

### Whisper evaluation

Runs Whisper on each test split, computes **WER**, **CER**, and **RTF** per file, writes detailed CSVs for raw and watermarked data, and a side-by-side **summary** including average RTF.

---

## Behaviour notes

### Audio loading

- **WAV**, **FLAC**, and **MP3** are supported (implementation uses PySoundFile, librosa for MP3 where needed).
- Non-WAV formats may be converted internally for processing.

### Tensor shapes

- Algorithms expect `(batch, channels, samples)`; mono/stereo handling is backend-specific.

### PyTorch compilation

- `TORCHDYNAMO_DISABLE=1` is set to avoid MSVC requirements on Windows (see `main.py`).

### Sample rates

- Watermark paths often assume ~16 kHz where documented in code; Whisper evaluation resamples to `--sample-rate`.

---

## References

- AudioSeal: [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)
- SilentCipher: [SesameAILabs/silentcipher](https://github.com/SesameAILabs/silentcipher)
- WavMark: [watermarking via adversarial examples](https://github.com/watermarking)
- Whisper: [openai/whisper](https://github.com/openai/whisper)

## License

Scripts for research use; algorithms follow their respective upstream licenses.
