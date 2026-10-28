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

For LLM post-correction (`--correct`, Apple Silicon only) install the optional extra:

```bash
poetry install --no-root --extras correct
```

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
| `--correct` | Proofread the transcript with a small local LLM. The raw transcript is kept as `<output>.raw.txt`. |
| `--llm-model` | mlx-lm model for `--correct`. Default: `mlx-community/Qwen2.5-3B-Instruct-4bit`. |

## Language detection

When `--language` is omitted, Whisper identifies the language from the audio itself
(the first ~30 seconds). The detected language and its confidence are printed to stderr.
For recordings that mix several languages, detection picks one, so pass `-l` explicitly.

## Notes

- The model is downloaded automatically on first use.
- Larger models (`medium`, `large-v3`) are more accurate, especially for non-English
  languages, but slower.
- A voice-activity filter is enabled to reduce hallucinations on silent sections.

## Progress

While transcribing, every line is prefixed with the percentage of the file processed,
e.g. `[ 40%] some text`. During `--correct`, progress is shown per chunk of segments.

## LLM correction

`--correct` runs a local instruct model through [mlx-lm](https://github.com/ml-explore/mlx-lm)
after transcription (Apple Silicon only). The model is chosen per language in `corrector.py`:

| Language | Default model | Download |
| --- | --- | --- |
| `pl` | `speakleash/Bielik-11B-v2.2-Instruct-MLX-4bit` (Polish-native) | ~6 GB |
| everything else | `mlx-community/Qwen2.5-3B-Instruct-4bit` (multilingual) | ~2 GB |

Override with `--llm-model`, or add entries to `LANGUAGE_MODELS` in `corrector.py`.

The model only fixes clear recognition errors, punctuation and capitalization. Safeguards:

- Timestamps are never sent to the model.
- If the model changes the number of lines or rewrites a line too much, the original line is kept.
- Whisper segments are cut at arbitrary points, so models tend to treat each line as a full sentence.
  The prompt forbids this, and `keep_line_boundaries()` additionally restores each line's original
  final punctuation and first-letter case. As a side effect, genuinely wrong line-final punctuation is not fixed.

The model fixes many misheard words but cannot guess names or jargon it never saw
(e.g. a misrecognised author's name), and 4-bit models still make mistakes. Review the result.

### Correcting an existing transcript

```bash
poetry run python correct.py transcript.txt -l pl              # writes transcript.corrected.txt
poetry run python correct.py transcript.txt -l pl --lines 48   # quick preview on the first 48 lines
```

Correction is slow on large files (a lecture of ~1900 lines is ~240 chunks, roughly an hour with Bielik 11B).
