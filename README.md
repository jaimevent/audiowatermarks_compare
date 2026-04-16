# audiowatermarks_compare

Small Python utility to run **Meta [AudioSeal](https://github.com/facebookresearch/audioseal)** on a folder of audio files: embed a watermark with the generator, then score it with the detector. Intended as a starting point for comparing or benchmarking audio watermarking setups.

## Requirements

- **Python** 3.10 or newer (aligned with PyTorch 2.11 and AudioSeal).
- **PyTorch**, **torchaudio**, **torchcodec**, **audioseal**, and **soundfile** (see `requirements.txt`).

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

Point the script at a directory that contains **`.wav`** and/or **`.flac`** files (non-recursive: only the top level of that folder is scanned):

```bash
python main.py path\to\audio_folder
```

Example:

```bash
python main.py .\dataset\lld\
```

For each file, the script loads audio, adds a batch dimension, runs `audioseal_wm_16bits`, adds the watermark to the waveform, runs `audioseal_detector_16bits`, and prints a line with the file name plus detector outputs.

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
