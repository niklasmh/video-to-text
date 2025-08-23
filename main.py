from argparse import ArgumentParser
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

def spinner_task(description):
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold]{description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(description, total=None)
        return task, progress

supported_languages = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su', 'yue', 'my', 'ca', 'nl', 'ht', 'lb', 'ps', 'pa', 'ro', 'ro', 'si', 'es', 'zh']

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the video file")
    parser.add_argument("name", type=str, nargs="?", help="Optional output name. Using input filename by default.")
    parser.add_argument("-m", "--model", type=str, default="openai/whisper-large-v3", help="Hugging Face model name. Using `openai/whisper-large-v3` by default.")
    parser.add_argument("-l", "--language", type=str, default="en", help="Language code for transcription. Using `en` by default.")
    args = parser.parse_args()

    if args.language not in supported_languages:
        console.print(f"Unsupported language: {args.language}. Supported languages are: {supported_languages}")
        raise SystemExit(1)

    VIDEO_FILE = args.file
    if not os.path.exists(VIDEO_FILE):
        raise FileNotFoundError(f"File not found: {VIDEO_FILE}")

    FILENAME = args.name if args.name else ".".join(VIDEO_FILE.split(".")[:-1])
    AUDIO_FILE = FILENAME + ".mp3"

    with Progress(SpinnerColumn(), TextColumn(" [bold green]Converting video to audio..."), transient=True, console=console) as progress:
        task = progress.add_task("convert", total=None)
        os.system(f"ffmpeg -y -hide_banner -loglevel error -i {VIDEO_FILE} {AUDIO_FILE}")
    console.print(":white_check_mark: [green]Audio conversion done![/green]")

    with Progress(SpinnerColumn(), TextColumn(" [bold green]Loading ASR model..."), transient=True, console=console) as progress:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        task = progress.add_task("load_model", total=None)
        model_id = args.model
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
    console.print(":white_check_mark: [green]ASR model loaded![/green]")

    with Progress(SpinnerColumn(), TextColumn(" [bold green]Loading audio file..."), transient=True, console=console) as progress:
        import librosa
        task = progress.add_task("load_audio", total=None)
        audio, sr = librosa.load(AUDIO_FILE, sr=16000)
    console.print(":white_check_mark: [green]Audio loaded![/green]")

    max_length = 30 * sr
    transcriptions = []

    with open(f"transcription.{FILENAME}.txt", "w") as f:
        f.write("")

    console.print(f":hourglass_flowing_sand: [bold green]Starting transcription...[/bold green] This may take a while depending on the audio length and model size.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Transcribing audio...", total=len(audio)//max_length+1)
        for start in range(0, len(audio), max_length):
            chunk = audio[start:start+max_length]

            if len(chunk) == 0:
                progress.advance(task)
                continue

            inputs = processor(
                torch.tensor(chunk),
                sampling_rate=sr,
                return_tensors="pt",
                return_attention_mask=True,
            )
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    task="transcribe",
                    language=args.language,
                )

            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(text)

            with open(f"transcription.{FILENAME}.txt", "a") as f:
                f.write(text.strip() + "\n\n")

            progress.advance(task)

    console.print(":white_check_mark: [bold green]Transcription complete![/bold green]")
    console.print(f"You can find the results in [yellow]transcription.{FILENAME}.txt[/yellow]")

    with open(f"transcription-full.{FILENAME}.txt", "w") as f:
        f.write("".join(transcriptions))
