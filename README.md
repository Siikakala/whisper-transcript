# Local Audio Transcription with Speaker Diarization (Apple Silicon)

A local Python tool for transcribing long audio files into speaker-labeled Markdown (`.md`) documents on Apple Silicon Macs (M1/M2/M3/M4).

It combines **Pyannote Audio** (running on PyTorch MPS) to detect who is speaking and when, with **MLX-Whisper** (`whisper-large-v3-mlx`) for fast, hardware-accelerated speech-to-text transcription.

[!NOTE] This script was vibe coded! It was built iteratively through AI-assisted prompting to solve real-world macOS dependency quirks, long-file caching, and local Apple Silicon acceleration.

## Features

- **100% Local Execution:** No audio data leaves your Mac.
- **Apple Silicon Optimized:** Uses PyTorch `mps` for speaker diarization and Apple's `mlx` framework for Whisper inference.
- **Diarization Caching (`.rttm`):** Automatically saves the speaker timeline to a `.rttm` file next to your audio. If transcription is interrupted, re-running skips the diarization step. Removed automatically if `.md` file is created successfully.
- **Batch Processing:** Supports multiple files and shell wildcards (`*.mp3`, `*.m4a`, `*.wav`), automatically skipping files that already have a `.md` transcript.
- **Configurable Language:** Set your target language (`fi`, `en`, `sv`, etc.) or `None` for automatic detection at the top of `transcribe.py`.

---

## Prerequisites

1. **Apple Silicon Mac** (M1, M2, M3, or M4) running macOS.
2. **Homebrew** and **FFmpeg**:
   ```bash
   brew install ffmpeg
   ```
3. **Conda / Miniconda** (recommended to isolate Python 3.11 and PyTorch dependencies).
   ```bash
   brew install --cask miniconda
   ```

---

## Environment Setup

Clone this repository and enter it
```bash
git clone https://github.com/Siikakala/whisper-transcript.git
cd whisper-transcript
```

Create a dedicated Conda environment with **Python 3.11** (newer Python versions like 3.13+ or 3.15 may have compatibility issues with `pyannote.audio` dependencies):

```bash
conda create -n transcribe python=3.11 -y
conda activate transcribe
```

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Note: `librosa` is used to decode audio directly into memory tensors, bypassing `torchcodec` dynamic library conflicts on macOS).*

---

## Hugging Face & Gated Models Setup

Pyannote's speaker diarization models are open-weights, but they are gated on Hugging Face and require accepting their user conditions before downloading:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Visit **all three** model pages below while logged in and click **"Agree and access repository"** on each:
   - [pyannote/speaker-diarization-3.1](https://huggingface.copyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.copyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://huggingface.copyannote/speaker-diarization-community-1)
3. Generate an Access Token (with `Read` permissions) at [huggingface.co/settings/tokens](https://huggingface.cosettings/tokens).

Token is primarily used from environmental variable but can also be added to the script configuration. If both are configured, environmental variable is preferred.

*(Note: On your first run, `mlx-whisper` will also automatically download `mlx-community/whisper-large-v3-mlx` (~3 GB) to your local Hugging Face cache).*

---

## Configuration

Open `transcribe.py` and adjust the top configuration block if needed:

```python
# --- CONFIGURATION ---
FALLBACK_HF_TOKEN = "" # Alternative Hugging Face token
MODEL_REPO = "mlx-community/whisper-large-v3-mlx"
LANGUAGE = "fi"  # ISO 639-1 language code (e.g., "fi", "en", "sv") or None for auto-detect
# ---------------------
```

---

## Usage

1. Activate your Conda environment and set your Hugging Face token:
   ```bash
   conda activate transcribe
   export HF_TOKEN="your_token_here"
   ```

2. **Transcribe a single file:**
   ```bash
   python transcribe.py ~/recording.mp3
   ```

3. **Batch transcribe multiple files using wildcards:**
   ```bash
   python transcribe.py ~/*.mp3
   ```

4. **Prevent your Mac from sleeping during long overnight batches:**
   ```bash
   caffeinate -i python transcribe.py ~/*.mp3
   ```

5. **Force re-processing (ignore existing `.md` and `.rttm` files):**
   ```bash
   python transcribe.py ~/recording.mp3 --overwrite
   ```

6. **Retain `.rttm` cache files:**
```bash
   python transcribe.py ~/recording.mp3 --preserve-cache
   ```

## Output Format

For an input file named `session.mp3`, the script generates:
- `session.rttm` — Cached speaker timeline timestamps. Removed after transcription (unless `--preserve-cache` is used)
- `session.md` — Formatted Markdown transcript:

```markdown
# Transkriptio: session.mp3

**[00:02] SPEAKER_05:** Sarbreen oli siis paikka, minne teidät neuvottiin menemään.

**[00:23] SPEAKER_00:** Hyvin.

**[00:27] SPEAKER_04:** Minä en ollut joo.
```

