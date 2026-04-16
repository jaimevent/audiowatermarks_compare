#!/usr/bin/python

import os

# AudioSeal wraps the encoder with torch.compile; Inductor on Windows needs MSVC (cl.exe).
# Without Build Tools, compilation fails. Disable Dynamo before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
import tempfile
import torch

torch._dynamo.config.disable = True  # belt-and-suspenders if env above is ignored

import soundfile as sf
from audioseal import AudioSeal


def convert_flac_to_wav(src_flac: str, dst_wav: str) -> None:
    """Decode FLAC and write a WAV container (float32 samples). libsndfile handles FLAC."""
    data, samplerate = sf.read(src_flac, dtype="float32", always_2d=True)
    sf.write(dst_wav, data, samplerate, format="WAV", subtype="FLOAT")


def _load_from_wav_file(wav_path: str) -> tuple[torch.Tensor, int]:
    # soundfile avoids torchaudio 2.10+ routing through torchcodec (FFmpeg DLLs on Windows).
    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy())
    return wav, sample_rate


def load_audio(file_path: str) -> tuple[torch.Tensor, int]:
    """Load audio for the model. WAV is read directly; FLAC is converted to WAV then loaded."""
    if file_path.lower().endswith(".flac"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            convert_flac_to_wav(file_path, tmp_path)
            return _load_from_wav_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    return _load_from_wav_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <audio_folder>")
        sys.exit(1)
    
    # Load the model
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()
    
    audio_folder = sys.argv[1]
    audio_files = [
        f
        for f in os.listdir(audio_folder)
        if f.lower().endswith(".wav") or f.lower().endswith(".flac")
    ]
    for audio_file in audio_files:
        audio_path = os.path.join(audio_folder, audio_file)
        wav, sample_rate = load_audio(audio_path)
        # Model expects (batch, channels, samples); file loaders return (channels, samples).
        wav = wav.unsqueeze(0)
        watermark = model.get_watermark(wav)
        watermarked_audio = wav + watermark

        detector = AudioSeal.load_detector("audioseal_detector_16bits")
        result, message = detector.detect_watermark(watermarked_audio)
        
        print(f"Audio: {audio_file}, Result: {result[:, 1 , :]}, Message: {message}")

# Other way is to load directly from the checkpoint
# model =  Watermarker.from_pretrained(checkpoint_path, device = wav.device)

# a torch tensor of shape (batch, channels, samples) and a sample rate
# It is important to process the audio to the same sample rate as the model
# expects. The default AudioSeal should work well with 16kHz and 24kHz, and 
# in the case of 48 khZ, it should work well for most speech audios
# wav = [load audio wav into a tensor of BatchxChannelxTime]

# watermark = model.get_watermark(wav)

# Optional: you can add a 16-bit message to embed in the watermark
# msg = torch.randint(0, 2, (wav.shape(0), model.msg_processor.nbits), device=wav.device)
# watermark = model.get_watermark(wav, message = msg)

# watermarked_audio = wav + watermark

# detector = AudioSeal.load_detector("audioseal_detector_16bits")

# To detect the messages in the high-level.
# result, message = detector.detect_watermark(watermarked_audio)

# print(result) # result is a float number indicating the probability of the audio being watermarked,
# print(message)  # message is a binary vector of 16 bits

# To detect the messages in the low-level.
# result, message = detector(watermarked_audio)

# result is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
# A watermarked audio should have result[:, 1, :] > 0.5
# print(result[:, 1 , :])  

# Message is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
# message will be a random tensor if the detector detects no watermarking from the audio
# print(message)