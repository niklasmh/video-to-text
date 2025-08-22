import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the video file")
    parser.add_argument("name", type=str, nargs="?", help="Optional output name. Using input filename by default.")
    parser.add_argument("-m", "--model", type=str, default="NbAiLabBeta/nb-whisper-medium", help="Hugging Face model name. Using a norwegian model by default.")
    args = parser.parse_args()

    VIDEO_FILE = args.file
    if not os.path.exists(VIDEO_FILE):
        raise FileNotFoundError(f"File not found: {VIDEO_FILE}")

    FILENAME = args.name if args.name else ".".join(VIDEO_FILE.split(".")[:-1])
    AUDIO_FILE = FILENAME + ".mp3"

    print(f"Converting {VIDEO_FILE} to {AUDIO_FILE}...")
    os.system(f"ffmpeg -y -hide_banner -loglevel error -i {VIDEO_FILE} {AUDIO_FILE}")
    print(f"Done!")

    # Load model + processor
    print(f"Loading model and processor...")
    model_id = args.model
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Load audio (mono, 16kHz)
    print(f"Loading audio from {AUDIO_FILE}...")
    audio, sr = librosa.load(AUDIO_FILE, sr=16000)

    # Whisper input window is ~30s → 30 * 16000 = 480000 samples
    max_length = 30 * sr

    transcriptions = []

    with open(f"transcription.{FILENAME}.txt", "w") as f:
        f.write("")

    print()
    print()

    # Split into 30s chunks
    print(f"Starting transcription...")
    for start in tqdm(range(0, len(audio), max_length)):
        chunk = audio[start:start+max_length]

        # Skip empty chunks
        if len(chunk) == 0:
            continue

        # Preprocess
        inputs = processor(
            torch.tensor(chunk),
            sampling_rate=sr,
            return_tensors="pt"
        )

        # Get decoder prompt (Norwegian, transcribe)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            task="transcribe",
            language="no"
        )

        # Run inference
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids
            )

        # Decode
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(text)

        with open(f"transcription.{FILENAME}.txt", "a") as f:
            f.write(text.strip() + "\n\n")

    print(f"Transcription complete! You can find the results in transcription.{FILENAME}.txt")

    with open(f"transcription-full.{FILENAME}.txt", "w") as f:
        f.write("".join(transcriptions))
