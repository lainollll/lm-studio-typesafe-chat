"""Launched with NC's interpreter. JSON lines over pipes; no NC app startup or edits."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import io
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace


def speech_chunks(text: str) -> list[str]:
    """Preserve prosody cues; prefer sentences or substantial clauses over word cuts."""
    chunks: list[str] = []
    words: list[str] = []
    abbreviations = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "e.g.", "i.e."}
    for word in text.split():
        words.append(word)
        bare = word.rstrip('"\u201d\u2019\x27)')
        abbreviation = bare.lower() in abbreviations or bool(re.fullmatch(r'(?:[A-Za-z]\.)+', bare))
        sentence_end = bare.endswith((".", "!", "?", "\u2026")) and not abbreviation
        clause_end = len(words) >= 5 and bare.endswith((",", ";", ":", "\u2014"))
        if len(words) >= 24 or sentence_end or clause_end:
            chunks.append(" ".join(words))
            words = []
    if words:
        chunks.append(" ".join(words))
    return chunks


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    device = sys.argv[2]
    sys.path.insert(0, str(root))
    protocol = sys.stdout

    def send(payload: dict) -> None:
        protocol.write(json.dumps(payload) + "\n")
        protocol.flush()

    try:
        with redirect_stdout(sys.stderr):
            import torch
            import soundfile as sf
            if "--local" in sys.argv[3:]:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
                service = ChatterboxTurboTTS.from_pretrained(device=selected_device)
            else:
                from addons.chatterbox_tts.service import ChatterboxTTSService

                class RuntimeConfig:
                    def get(self, key, default=None):
                        return default

                    def tts_device(self):
                        return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

                context = SimpleNamespace(capabilities=SimpleNamespace(
                    runtime_config=RuntimeConfig(), diagnostics=lambda *args, **kwargs: None))
                service = ChatterboxTTSService(context)
                service._ensure_model()
        send({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                text = str(request["text"])
                if not text or len(text) > 4000:
                    raise ValueError("Invalid narration length")
                chunks = speech_chunks(text) if request.get("stream") else [text]
                for chunk in chunks:
                    with redirect_stdout(sys.stderr):
                        waveform = service.generate(chunk, audio_prompt_path=request.get("reference") or None)
                        samples = waveform.detach().cpu().numpy().squeeze()
                        stream = io.BytesIO()
                        sf.write(stream, samples, service.sr, format="WAV", subtype="PCM_16")
                    send({"audio": base64.b64encode(stream.getvalue()).decode("ascii")})
                if request.get("stream"):
                    send({"done": True})
            except Exception as exc:
                send({"error": f"Chatterbox synthesis failed ({type(exc).__name__})."})
    except Exception as exc:
        send({"error": f"Chatterbox could not load ({type(exc).__name__}). Check the selected installation and cached Turbo weights."})


if __name__ == "__main__":
    main()
