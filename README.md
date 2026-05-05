# Audiowatermarks Compare

Python utility to benchmark audio watermarking algorithms on folders of audio files. Supports embedding watermarks, detection, robustness testing against attacks, and evaluating impact on ASR models like DeepSpeech.

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

### DeepSpeech (for `evaluate` command)

The `evaluate` command requires OpenAI Whisper for evaluating WER/CER on raw and watermarked datasets.

Whisper is available on PyPI:

```bash
pip install openai-whisper
```

Whisper will automatically download the model weights on first use.

## Usage

Point the script at a directory containing **`.wav`**, **`.flac`**, or **`.mp3`** files. Only the **top level** of that folder is scanned (no subfolders).

### Commands

- **`watermark`**: Embed watermarks and measure quality metrics.
- **`detect`**: Run watermark detection on audio files.
- **`attack`**: Embed watermarks, apply attacks, then detect (robustness testing).
- **`evaluate`**: Evaluate WER/CER using OpenAI Whisper on raw and watermarked datasets.

### Examples

Embed watermarks with default AudioSeal:
```bash
python main.py watermark -i path/to/audio_folder
```

Detect watermarks:
```bash
python main.py detect -i path/to/audio_folder
```

Test robustness against attacks:
```bash
python main.py attack -i path/to/audio_folder
```

Train and evaluate DeepSpeech models:
```bash
python main.py evaluate --raw-dataset /path/to/raw/csv --watermarked-dataset /path/to/watermarked/csv
```

### Command-line options

#### Global options
| Option | Description |
|--------|-------------|
| `-i`, `--input` | **Required for watermark/detect/attack.** Folder containing `.wav`/`.flac`/`.mp3`. |
| `--algorithm` | Algorithm to use (default: `audioseal`). Choices: `audioseal`, `silentcipher`, `wavmark`. |
| `--all-algorithms` | Run all supported algorithms. |

#### Watermark command
| Option | Description |
|--------|-------------|
| `-o`, `--output-plot` | Directory for PNG plots (default: `<input>/plots`). |
| `--no-plots` | Skip comparison plots. |
| `--plot-dpi` | PNG resolution (default: `150`). |
| `--output-watermarked` | Directory for watermarked WAV files (default: `<input>/<algorithm>_watermarked_wav`). |
| `--output-metrics` | Base name for metrics CSV (default: `<algorithm>_watermark_metrics.csv`). |

#### Detect command
| Option | Description |
|--------|-------------|
| `--detection-threshold` | Frame-level P(watermark) threshold (default: `0.5`). |
| `--message-threshold` | Message bit threshold (default: `0.5`). |
| `--file-fraction-threshold` | Fraction of frames above threshold to count as detected (default: `0.5`). |
| `--output-detect-log` | Base name for detection log CSV. |

#### Attack command
Includes all watermark and detect options, plus:
| Option | Description |
|--------|-------------|
| `--attack-seed` | RNG seed for stochastic attacks (default: `42`). |
| `--save-attacked` | Directory to save attacked watermarked WAVs. |
| `--output-attack-metrics` | Base name for attack metrics CSV. |
| `--output-attack-plot` | Directory for attack robustness plots. |
| `--no-attack-plots` | Skip attack summary plots. |

#### Evaluate command
| Option | Description |
|--------|-------------|
| `--raw-dataset` | **Required.** Directory with raw dataset CSV/TSV files. |
| `--watermarked-dataset` | **Required.** Directory with watermarked dataset CSV/TSV files. |
| `--output-root` | Output directory for results (default: `whisper_results`). |
| `--model-size` | Whisper model size: tiny/base/small/medium/large (default: `base`). |
| `--sample-rate` | Sample rate for evaluation (default: `16000`). |

## What the script does

### Watermark embedding
1. Load audio (WAV directly; FLAC/MP3 converted to temporary WAV).
2. Embed watermark and compute `watermarked = original + watermark`.
3. Print SNR (dB) — original power vs residual power.
4. Save watermarked WAV.
5. Compute BER/NC metrics.
6. Optionally save comparison plots (waveforms and spectrograms).

### Detection
- Run detector on audio files.
- Log detection results to CSV.

### Attacks
- Embed watermarks.
- Apply various audio attacks (resampling, compression, noise, etc.).
- Detect watermarks on attacked audio.
- Generate robustness heatmaps and bar charts.

### Whisper evaluation
- Evaluate WER/CER on the test sets of raw and watermarked datasets using pre-trained Whisper models.
- Output comparison summary.

## Behaviour notes

### Audio loading
- **WAV**, **FLAC**, and **MP3** supported via PySoundFile and librosa.
- Non-WAV formats converted to temporary WAV before processing.

### Tensor shapes
- Algorithms expect `(batch, channels, samples)`.
- Mono audio converted to stereo if needed.

### PyTorch compilation
- `TORCHDYNAMO_DISABLE=1` set to avoid MSVC requirements on Windows.

### Sample rates
- Algorithms work best at 16 kHz; resample if needed.

## References

- AudioSeal: [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)
- SilentCipher: [SesameAILabs/silentcipher](https://github.com/SesameAILabs/silentcipher)
- WavMark: [watermarking via adversarial examples](https://github.com/watermarking)
- Whisper: [openai/whisper](https://github.com/openai/whisper)

## License

Scripts for research use; algorithms follow their respective upstream licenses.
