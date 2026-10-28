#!/usr/bin/env python3
"""Proofread an existing transcript (one segment per line) with a small local LLM.

Usage:
    python correct.py transcript.txt -l pl
    python correct.py transcript.txt -l pl -o fixed.txt --lines 80
"""
import argparse
import sys
from pathlib import Path

from corrector import Corrector


def main() -> int:
    parser = argparse.ArgumentParser(description="Proofread a transcript with a small local LLM.")
    parser.add_argument("transcript", type=Path, help="text file, one segment per line")
    parser.add_argument("-l", "--language", help="language code, e.g. pl, en (default: let the model infer)")
    parser.add_argument("-o", "--output", type=Path, help="output file (default: <transcript>.corrected.txt)")
    parser.add_argument("--llm-model", help="mlx-lm model (default: see corrector.py)")
    parser.add_argument("--lines", type=int, help="only correct the first N lines (for a quick preview)")
    args = parser.parse_args()

    if not args.transcript.is_file():
        print(f"Error: file not found: {args.transcript}", file=sys.stderr)
        return 1

    lines = args.transcript.read_text(encoding="utf-8").splitlines()
    if args.lines:
        lines = lines[: args.lines]
    output = args.output or args.transcript.with_suffix(".corrected.txt")

    corrector = Corrector(args.language, args.llm_model)

    def report(done: int, total: int) -> None:
        print(f"[{done * 100 // total:3d}%] Correcting: chunk {done}/{total}", file=sys.stderr)

    fixed = corrector.correct(lines, on_progress=report)
    output.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    print(f"Corrected transcript saved to: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
