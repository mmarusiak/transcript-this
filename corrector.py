"""Post-correction of Whisper transcripts with a small local LLM (mlx-lm, Apple Silicon)."""
import sys
from typing import Callable, Optional

# One multilingual instruct model covers Polish, English and many other languages.
# Override with --llm-model, or pass a per-language mapping via LANGUAGE_MODELS.
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
LANGUAGE_MODELS: dict[str, str] = {}

LANGUAGE_NAMES = {
    "pl": "Polish", "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "cs": "Czech", "sk": "Slovak",
    "uk": "Ukrainian", "ru": "Russian", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "tr": "Turkish", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
}

SYSTEM_PROMPT = (
    "You are a proofreader for automatic speech-recognition transcripts. "
    "The text is in {language}. Fix only clear recognition errors: misspelled or misheard words, "
    "wrong word forms, punctuation and capitalization. Do NOT paraphrase, summarize, translate, "
    "add or remove information. The input has one segment per line; return exactly the same "
    "number of lines in the same order, with no numbering, comments or extra text."
)

CHUNK_SEGMENTS = 8
MIN_RATIO, MAX_RATIO = 0.6, 1.5


class Corrector:
    def __init__(self, language: Optional[str], model_name: Optional[str] = None):
        try:
            from mlx_lm import load
        except ImportError:
            raise SystemExit(
                "Correction needs the optional dependency (Apple Silicon only): "
                "poetry install --no-root --extras correct"
            )
        self.language = LANGUAGE_NAMES.get(language or "", language or "the original language")
        name = model_name or LANGUAGE_MODELS.get(language or "", DEFAULT_MODEL)
        print(f"Loading correction model '{name}'...", file=sys.stderr)
        self.model, self.tokenizer = load(name)

    def _fix_chunk(self, lines: list[str]) -> list[str]:
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(language=self.language)},
            {"role": "user", "content": "\n".join(lines)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        max_tokens = max(256, int(len(prompt) * 0.5))
        out = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        fixed = [l.strip() for l in out.strip().splitlines() if l.strip()]

        # Safety net: if the model changed the structure or rewrote too much, keep the original.
        if len(fixed) != len(lines):
            return lines
        result = []
        for orig, new in zip(lines, fixed):
            ratio = len(new) / max(len(orig), 1)
            result.append(new if MIN_RATIO <= ratio <= MAX_RATIO else orig)
        return result

    def correct(self, lines: list[str], on_progress: Optional[Callable[[int, int], None]] = None) -> list[str]:
        chunks = [lines[i:i + CHUNK_SEGMENTS] for i in range(0, len(lines), CHUNK_SEGMENTS)]
        result: list[str] = []
        for n, chunk in enumerate(chunks, 1):
            result.extend(self._fix_chunk(chunk))
            if on_progress:
                on_progress(n, len(chunks))
        return result
