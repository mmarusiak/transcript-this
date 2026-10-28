#!/usr/bin/env python3
"""Transcribe an audio recording (any format) to a text file.

Usage:
    python transcribe.py recording.m4a
    python transcribe.py recording.mp3 -l pl -o result.txt
    python transcribe.py recording.wav --model medium --timestamps

If no language is given, it is detected automatically by the Whisper model
from the audio itself (the first ~30 seconds of the recording).
"""
import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def format_ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio to a text file.")
    parser.add_argument("audio", type=Path, help="path to the recording (any format supported by ffmpeg/PyAV)")
    parser.add_argument("-l", "--language", help="language code, e.g. pl, en, de (default: auto-detect)")
    parser.add_argument("-o", "--output", type=Path, help="output file (default: <recording>.txt)")
    parser.add_argument(
        "-m", "--model", default="small",
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: small)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    parser.add_argument("--timestamps", action="store_true", help="prefix each segment with a timestamp")
    args = parser.parse_args()

    if not args.audio.is_file():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        return 1

    output = args.output or args.audio.with_suffix(".txt")

    print(f"Loading model '{args.model}'...", file=sys.stderr)
    model = WhisperModel(args.model, device=args.device, compute_type="auto")

    segments, info = model.transcribe(
        str(args.audio),
        language=args.language,  # None => auto-detect from the audio
        vad_filter=True,
    )

    if args.language:
        print(f"Language (specified): {args.language}", file=sys.stderr)
    else:
        print(
            f"Detected language: {info.language} (confidence {info.language_probability:.0%})",
            file=sys.stderr,
        )

    with output.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.text.strip()
            line = f"[{format_ts(seg.start)}] {text}" if args.timestamps else text
            f.write(line + "\n")
            print(line, file=sys.stderr)
            f.flush()

    print(f"\nTranscript saved to: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
