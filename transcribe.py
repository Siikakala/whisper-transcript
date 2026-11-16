import os
# Disable ugly Hugging Face Hub cache-check bars before importing huggingface_hub dependents
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

import argparse
import glob
import time
import torch
import librosa
import mlx_whisper
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database.util import load_rttm

# --- CONFIGURATION ---
FALLBACK_HF_TOKEN = "" # Alternative Hugging Face token
MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
LANGUAGE = "fi"  # ISO 639-1 language code (e.g., "fi", "en", "sv") or None for auto-detect
# ---------------------


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours > 0 else f"{minutes:02d}:{secs:02d}"


def process_file(input_path: str, pipeline: Pipeline, overwrite: bool = False, preserve_cache: bool = False):
    input_path = os.path.expanduser(input_path)
    if not os.path.exists(input_path):
        print(f"⚠️  Skipping: File '{input_path}' not found.")
        return

    base_name, _ = os.path.splitext(input_path)
    output_md, cache_rttm = f"{base_name}.md", f"{base_name}.rttm"

    if os.path.exists(output_md) and not overwrite:
        print(f"⏭️  Skipping '{os.path.basename(input_path)}' -> '{os.path.basename(output_md)}' already exists.")
        return

    file_start_time = time.time()
    print(f"🎵 Loading audio: {os.path.basename(input_path)}")

    audio_np, sample_rate = librosa.load(input_path, sr=16000, mono=True)
    print(f"⏱️  Audio duration: {format_time(len(audio_np) / sample_rate)}")

    if os.path.exists(cache_rttm) and not overwrite:
        print(f"⚡ Found cached diarization '{os.path.basename(cache_rttm)}', skipping Pyannote step...")
        annotation = next(iter(load_rttm(cache_rttm).values()))
    else:
        waveform_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        print("🔍 Analyzing speaker turns (Diarization)...")
        with ProgressHook() as hook:
            diarization_output = pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate}, hook=hook)
        annotation = getattr(diarization_output, "speaker_diarization", diarization_output)
        with open(cache_rttm, "w", encoding="utf-8") as rttm_file:
            annotation.write_rttm(rttm_file)
        print(f"💾 Cached speaker timeline to '{os.path.basename(cache_rttm)}'")

    valid_turns = [(turn, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True) if (turn.end - turn.start) >= 0.3]
    total_segments = len(valid_turns)
    print(f"🤖 Transcribing {total_segments} speaker segments with MLX-Whisper (language: {LANGUAGE or 'auto-detect'})...")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# Transkriptio: {os.path.basename(input_path)}\n\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, style="dim", complete_style="magenta", finished_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("• ETA:"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Transcribing", total=total_segments)

            for turn, speaker in valid_turns:
                chunk = audio_np[int(turn.start * sample_rate):int(turn.end * sample_rate)]
                result = mlx_whisper.transcribe(chunk, path_or_hf_repo=MODEL_REPO, language=LANGUAGE, task="transcribe")
                text = result["text"].strip()
                if text:
                    markdown_line = f"**[{format_time(turn.start)}] {speaker}:** {text}\n\n"
                    f.write(markdown_line)
                    f.flush()
                progress.advance(task)

    if os.path.exists(cache_rttm) and not preserve_cache:
        os.remove(cache_rttm)
        print(f"🧹 Removed temporary cache file '{os.path.basename(cache_rttm)}'")

    print(f"✅ Saved '{output_md}' in {format_time(time.time() - file_start_time)}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file(s) with speaker diarization to Markdown on Apple Silicon.")
    parser.add_argument("audio_files", nargs="+", help="Path(s) or wildcard pattern to input audio files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .md and .rttm cache files")
    parser.add_argument("--preserve-cache", action="store_true", help="Keep the intermediate .rttm diarization cache file")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or FALLBACK_HF_TOKEN
    if not hf_token:
        print("❌ Error: Hugging Face token not found. Set it with: export HF_TOKEN='your_token_here'")
        return

    expanded_files = []
    for pattern in args.audio_files:
        matches = glob.glob(os.path.expanduser(pattern))
        expanded_files.extend(sorted(matches) if matches else [pattern])

    print(f"📂 Found {len(expanded_files)} file(s) to process.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Loading Pyannote Diarization Pipeline on {device}...")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    pipeline.to(device)

    total_start = time.time()
    for file_idx, audio_file in enumerate(expanded_files, start=1):
        print(f"\n📌 File {file_idx} of {len(expanded_files)}")
        process_file(audio_file, pipeline, overwrite=args.overwrite, preserve_cache=args.preserve_cache)

    print(f"\n🎉 All done! Total elapsed time: {format_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()

