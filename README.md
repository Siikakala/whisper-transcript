# Whisper Transcript

Fast audio transcription with speaker diarization on Apple Silicon using **MLX-Whisper** and **Pyannote.audio**.

> [!IMPORTANT]
> This script was vibe coded to create a clean, practical local transcription workflow.

## Features

* 🚀 **Optimized for Apple Silicon**: Leverages `mlx-whisper` for high-speed transcription.
* 👥 **Speaker Diarization**: Uses `pyannote/speaker-diarization-3.1` to separate and label different speakers.
* 🔄 **Smart Process Handling**: Automatically caches speaker diarization timelines into a `.rttm` file and tracks progress in the `.md` output. If a transcription run is interrupted (or if only diarization finished), running the script again will reuse the cached `.rttm` timeline, skip already-transcribed segments, and pick up right where it left off. Additionally, completed files will be skipped (`.md` exists but `.rttm` does not)
* 🇫🇮 Configurable Language: Set your target language (fi, en, sv, etc.) or None for automatic detection at the top of transcribe.py. _(defaults to fi)_
* 🛡️ **Clean Interrupts**: Gracefully handles `Ctrl+C` (`KeyboardInterrupt`) without cluttering your terminal with tracebacks.

## Prerequisites

* macOS with Apple Silicon (M1/M2/M3/M4)
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
   ```bash
   brew install --cask miniconda
   conda init zsh
   ```
   _Restart your terminal before continuing so the conda can initialize correctly!_

## Setup & Installation

### 1. Create a Conda Environment

Create and activate a dedicated Conda environment with Python 3.11 (newer Python versions like 3.13+ or 3.15 may have compatibility issues with pyannote.audio dependencies):

```bash
conda create -n whisper-env python=3.11 -y
conda activate whisper-env
```

### 2. Clone the Repository

```bash
git clone https://github.com/Siikakala/whisper-transcript.git
cd whisper-transcript
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
_(Note: librosa is used to decode audio directly into memory tensors, bypassing torchcodec dynamic library conflicts on macOS)._

### 4. Hugging Face Setup (Required for Pyannote)

Pyannote's diarization pipeline requires authentication and accepted model licenses:

1. Create a [Hugging Face account](https://huggingface.co/) and generate an Access Token (`Settings > Access Tokens`).
2. Visit and accept the user conditions for the required gated models:
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   * [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Set your token as an environment variable in your terminal:

   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```
   _The token can be set to the script configuration if you prefer storing it there. If both are set, the environment variable is preferred_

## Usage

Run the transcription script on one or more audio files (supports wildcards):

```bash
python transcribe.py "path/to/audio/*.mp3"
```

### First Run & Hugging Face Progress Bars

By default, Hugging Face Hub progress bars are disabled in the script (`os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"`). They rely on `tqdm` rather than `rich`, which clashes with the rest of the terminal UI and makes console output ugly and potentially confusing during cache checks.

However, on your **very first run**, the script needs to download several gigabytes of Whisper and Pyannote models, and it may appear frozen while doing so. To temporarily restore the download progress bars for your first run, pass `0` inline when executing the script:

```bash
HF_HUB_DISABLE_PROGRESS_BARS=0 python transcribe.py "path/to/audio/test.mp3"
```

### Output Files & Location

* **Transcripts (`.md`)**: The final transcription is automatically saved directly in the **same directory as your input audio file** with the matching base name (e.g., `audio.mp3` generates `audio.md`).
* **Temporary Cache (`.rttm`)**: During processing, a temporary diarization cache (`.rttm`) is created in the same folder to speed up potential resumes. Unless `--preserve-cache` is specified, this file is automatically cleaned up upon successful completion.

### Options

* `--overwrite`: Ignore existing cache/markdown files and re-run both diarization and transcription from scratch.
* `--preserve-cache`: Keep the intermediate `.rttm` diarization cache file after a successful run instead of automatically deleting it.
