import warnings
# Match by module name so multi-line torchcodec warnings are cleanly suppressed
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

import argparse
import glob
import os
import time
import torch
import librosa
import mlx_whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database.util import load_rttm

# --- CONFIGURATION ---
FALLBACK_HF_TOKEN = ""
MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
LANGUAGE = "fi"  # ISO 639-1 language code (e.g., "fi", "en", "sv") or None for auto-detect
# ---------------------

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def process_file(input_path: str, pipeline: Pipeline, overwrite: bool = False):
    input_path = os.path.expanduser(input_path)
    if not os.path.exists(input_path):
        print(f"⚠️  Skipping: File '{input_path}' not found.")
        return

    base_name, _ = os.path.splitext(input_path)
    output_md = f"{base_name}.md"
    cache_rttm = f"{base_name}.rttm"

    if os.path.exists(output_md) and not overwrite:
        print(f"⏭️  Skipping '{os.path.basename(input_path)}' -> '{os.path.basename(output_md)}' already exists (use --overwrite to redo).")
        return

    file_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"🎵 Loading audio: {os.path.basename(input_path)}")
    
    # Load audio at 16kHz mono via librosa
    audio_np, sample_rate = librosa.load(input_path, sr=16000, mono=True)
    duration_sec = len(audio_np) / sample_rate
    print(f"⏱️  Audio duration: {format_time(duration_sec)}")

    # Check if we already cached diarization results for this file
    if os.path.exists(cache_rttm) and not overwrite:
        print(f"⚡ Found cached diarization '{os.path.basename(cache_rttm)}', skipping Pyannote step...")
        rttm_data = load_rttm(cache_rttm)
        annotation = next(iter(rttm_data.values()))
    else:
        waveform_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        audio_in_memory = {"waveform": waveform_tensor, "sample_rate": sample_rate}

        print("🔍 Analyzing speaker turns (Diarization)...")
        with ProgressHook() as hook:
            diarization_output = pipeline(audio_in_memory, hook=hook)

        # pyannote.audio 4.x returns a DiarizeOutput object; extract the Annotation
        annotation = getattr(diarization_output, "speaker_diarization", diarization_output)

        # Save diarization cache immediately so long runs never have to be repeated
        with open(cache_rttm, "w", encoding="utf-8") as rttm_file:
            annotation.write_rttm(rttm_file)
        print(f"💾 Cached speaker timeline to '{os.path.basename(cache_rttm)}'")

    # Filter out tiny noise bursts under 0.3 seconds
    valid_turns = [
        (turn, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
        if (turn.end - turn.start) >= 0.3
    ]

    total_segments = len(valid_turns)
    lang_label = LANGUAGE if LANGUAGE else "auto-detect"
    print(f"🤖 Transcribing {total_segments} speaker segments with MLX-Whisper (language: {lang_label})...")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# Transkriptio: {os.path.basename(input_path)}\n\n")

        for idx, (turn, speaker) in enumerate(valid_turns, start=1):
            start_time = turn.start
            end_time = turn.end

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            chunk = audio_np[start_sample:end_sample]

            result = mlx_whisper.transcribe(
                chunk,
                path_or_hf_repo=MODEL_REPO,
                language=LANGUAGE,
                task="transcribe"
            )

            text = result["text"].strip()
            str_start = format_time(start_time)

            if text:
                markdown_line = f"**[{str_start}] {speaker}:** {text}\n\n"
                print(f"[{idx}/{total_segments}] {markdown_line.strip()}")
                f.write(markdown_line)
                f.flush()
            else:
                print(f"[{idx}/{total_segments}] [{str_start}] {speaker}: (no speech detected)")

    elapsed = time.time() - file_start_time
    print(f"✅ Saved '{output_md}' in {format_time(elapsed)}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file(s) with speaker diarization to Markdown on Apple Silicon."
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="Path(s) or wildcard pattern to input audio files (e.g., ~/Movies/mono-*.mp3)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .md and .rttm cache files instead of skipping them"
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or FALLBACK_HF_TOKEN
    if not hf_token:
        print("❌ Error: Hugging Face token not found.")
        print("Set it in your terminal with: export HF_TOKEN='your_token_here'")
        return

    expanded_files = []
    for pattern in args.audio_files:
        matches = glob.glob(os.path.expanduser(pattern))
        if matches:
            expanded_files.extend(sorted(matches))
        else:
            expanded_files.append(pattern)

    print(f"📂 Found {len(expanded_files)} file(s) to process.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Loading Pyannote Diarization Pipeline on {device}...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    pipeline.to(device)

    total_start = time.time()
    for file_idx, audio_file in enumerate(expanded_files, start=1):
        print(f"\n📌 File {file_idx} of {len(expanded_files)}")
        process_file(audio_file, pipeline, overwrite=args.overwrite)

    print(f"\n🎉 All done! Total elapsed time: {format_time(time.time() - total_start)}")

if __name__ == "__main__":
    main()

