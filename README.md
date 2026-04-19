# audiowatermarks_compare

Small Python utility to run **Meta [AudioSeal](https://github.com/facebookresearch/audioseal)** on a folder of audio files: embed a watermark with the generator, measure how loud the residual is versus the original, save the watermarked waveform, optionally export comparison plots, and score the result with the detector. It is a practical starting point for comparing or benchmarking audio watermarking setups.

## Requirements

- **Python** 3.10 or newer (aligned with PyTorch 2.11 and AudioSeal).
- **PyTorch**, **torchaudio**, **torchcodec**, **audioseal**, **soundfile**, and **matplotlib** (see `requirements.txt`).

Install PyTorch from the [official install page](https://pytorch.org/get-started/locally/) if you need a specific CUDA build; then install the rest:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, if you use the CPU wheel index for `torchcodec`, match it to your PyTorch channel, for example:

```bash
pip install torchcodec --index-url=https://download.pytorch.org/whl/cpu
```

## Usage

Point the script at a directory that contains **`.wav`** and/or **`.flac`** files. Only the **top level** of that folder is scanned (no subfolders).

```bash
python main.py -i path\to\audio_folder
```

Short form:

```bash
python main.py --input .\dataset\lld\
```

### Command-line options

| Option | Description |
|--------|-------------|
| `-i`, `--input` | **Required.** Folder containing `.wav` / `.flac`. |
| `-o`, `--output-plot` | Directory for PNG plots (default: `<input>/audioseal_plots`). |
| `--no-plots` | Skip waveform and spectrogram comparison figures. |
| `--plot-dpi` | PNG resolution (default: `150`). |
| `--output-watermarked` | Directory for watermarked WAV files (default: `<input>/watermarked_wav`). |
| `--generator` | AudioSeal generator card (default: `audioseal_wm_16bits`). |
| `--detector` | AudioSeal detector card (default: `audioseal_detector_16bits`). |
| `--debug` | Verbose detector logging in the console (default: `True`). |

Stopping with **Ctrl+C** exits cleanly: a short message on stderr, matplotlib figures closed, CUDA cache cleared when applicable, and exit code **130** (no traceback).

## What the script does per file

1. **Load** the clip (WAV directly; FLAC via a temporary WAV using the same pipeline).
2. **Embed** the watermark and form `watermarked = original + watermark`.
3. **Print embedding SNR (dB)** — signal power of the original versus mean-square power of the residual `watermarked − original` (higher dB means a quieter watermark relative to the host audio).
4. **Save** `<stem>_watermarked.wav` under the watermarked output directory (float32 WAV).
5. **Run** the detector on the watermarked tensor (high- and low-level API when debug logging is enabled).
6. **Optionally save** a four-panel PNG (original vs watermarked waveform and spectrogram), matching the layout used in the upstream AudioSeal notebook examples.

If a **CUDA** device is available, the generator and tensors used in the loop are moved to the GPU for inference.

## Behaviour notes

### Audio loading

- **WAV** and **FLAC** are supported. Loading uses **PySoundFile** (`soundfile`) so file I/O does not depend on **torchaudio’s TorchCodec path**, which on Windows often needs a full **FFmpeg shared** install on `PATH`.
- FLAC files are decoded and passed through a **temporary WAV** on disk before the same read path as WAV.

### Tensor shape

AudioSeal expects waveforms shaped **`(batch, channels, samples)`**. The script adds the batch axis with `unsqueeze(0)` after load.

### `torch.compile` on Windows

AudioSeal’s encoder can be wrapped with **`torch.compile`**. The Inductor backend on CPU Windows looks for **MSVC (`cl.exe`)**; if it is missing, compilation fails. This repository sets **`TORCHDYNAMO_DISABLE=1`** and **`torch._dynamo.config.disable = True`** in `main.py` before heavy imports so runs work without Visual Studio Build Tools. To try compiled mode again after installing the **Desktop development with C++** workload, remove or adjust those lines.

### Sample rate

AudioSeal is documented to work well at **16 kHz** and **24 kHz**, and for **48 kHz** speech in many cases. Resample upstream if you need a strict sample rate for your experiments.

## References

- AudioSeal: [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)
- Paper: [Proactive Detection of Voice Cloning with Localized Watermarking](https://arxiv.org/abs/2401.17264)

## License

This repository’s scripts are for research use; **AudioSeal** and model weights follow the licenses described in the upstream project.
