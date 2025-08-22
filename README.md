# Video to Text

Want to download a video and get the transcription for it? This is 100% offline, use any speech recognition model from [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) that you like.

DISCLAIMER: I have only tested with the [`NbAiLab/nb-whisper-base`](https://huggingface.co/NbAiLab/nb-whisper-base) models so far. But the code is easy to fix with some vibe coding.

## How to use

```bash
pip install librosa torch torchaudio tqdm transformers

python main.py my-video.mp4 # -> transcription.my-video.txt

# usage: main.py [-h] [-m MODEL] file [name]
#
# positional arguments:
#   file               Path to the video file
#   name               Optional output name. Using input filename by default.
#
# options:
#   -h, --help         show this help message and exit
#   -m, --model MODEL  Hugging Face model name. Using a norwegian model by default.
```
