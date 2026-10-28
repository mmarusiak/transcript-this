# transcript-this

Transcribe an audio recording (any format) to a text file using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Installation

Requires Python 3.11–3.13 and [Poetry](https://python-poetry.org/).

```bash
poetry config virtualenvs.in-project true --local
poetry env use python3.13
poetry install --no-root
```

This creates a `.venv` in the project directory.

No separate ffmpeg install is needed: faster-whisper decodes audio via the bundled PyAV,
so mp3, m4a, wav, ogg, flac, mp4, etc. all work.

## Usage

```bash
poetry run python transcribe.py recording.m4a                      # auto-detect language, writes recording.txt
poetry run python transcribe.py recording.mp3 -l pl -o result.txt  # specify language and output file
poetry run python transcribe.py recording.wav -m medium --timestamps
```

| Option | Description |
| --- | --- |
| `audio` | Path to the recording (required). |
| `-l`, `--language` | Language code (e.g. `pl`, `en`, `de`). Default: auto-detect. |
| `-o`, `--output` | Output file. Default: `<recording>.txt` next to the input. |
| `-m`, `--model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`. Default: `small`. |
| `--device` | `auto`, `cpu` or `cuda`. Default: `auto`. |
| `--timestamps` | Prefix each line with an `[hh:mm:ss]` timestamp. |

## Language detection

When `--language` is omitted, Whisper identifies the language from the audio itself
(the first ~30 seconds). The detected language and its confidence are printed to stderr.
For recordings that mix several languages, detection picks one, so pass `-l` explicitly.

## Notes

- The model is downloaded automatically on first use.
- Larger models (`medium`, `large-v3`) are more accurate, especially for non-English
  languages, but slower.
- A voice-activity filter is enabled to reduce hallucinations on silent sections.
