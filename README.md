# Video to Text

Want to download a video and get the transcription for it? This is 100% offline, use any speech recognition model from [Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) that you like.

## How to use

```bash
# Download repo
git clone https://github.com/niklasmh/video-to-text.git

# Install dependencies
pip install librosa rich torch torchaudio transformers

# Transcribe your-video.mp4
python main.py your-video.mp4 --model NbAiLab/nb-whisper-medium
# -> Output: transcription.your-video.txt
```

## CLI

```
usage: main.py [-h] [-m MODEL] [-l LANGUAGE] file [name]

positional arguments:
  file                  Path to the video file
  name                  Optional output name. Using input filename by default.

options:
  -h, --help            show this help message and exit
  -m, --model MODEL     Hugging Face model name. Using `openai/whisper-large-v3` by default.
  -l, --language LANGUAGE
                        Language code for transcription. Using `en` by default.
```

_DISCLAIMER: I have only tested with the [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) and [`NbAiLab/nb-whisper-base`](https://huggingface.co/NbAiLab/nb-whisper-base) models so far. There are probably models that does not work. But the code should be easy to fix with some vibe coding._
