"""Optional Chatterbox HTTP client and cancellable in-memory WAV playback."""
from dataclasses import dataclass, field
import io
import json
from threading import Event
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class SpeechError(Exception):
    pass


@dataclass(frozen=True)
class ChatterboxConfig:
    endpoint: str = "http://localhost:8004/tts"
    voice: str = ""
    model: str = "tts-1"
    key: str = field(default="", repr=False)


def request_audio(config: ChatterboxConfig, text: str) -> bytes:
    url = config.endpoint.strip()
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise SpeechError("Enter a valid Chatterbox HTTP endpoint URL without embedded credentials.")
    if parsed.path.rstrip("/").endswith("/audio/speech"):
        payload = {"model": config.model, "input": text, "voice": config.voice or "default", "response_format": "wav"}
    else:
        payload = {"text": text, "output_format": "wav", "stream": False}
        if config.voice:
            payload.update(voice_mode="predefined", predefined_voice_id=config.voice)
    headers = {"Content-Type": "application/json"}
    if config.key:
        headers["Authorization"] = "Bearer " + config.key
    try:
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urlopen(request, timeout=15) as response:
            audio = response.read(10 * 1024 * 1024 + 1)
    except HTTPError as exc:
        exc.close()
        raise SpeechError(f"Chatterbox HTTP {exc.code}. Check its endpoint, voice and API key.") from None
    except (URLError, OSError, ValueError):
        raise SpeechError("Chatterbox connection failed or timed out.") from None
    if len(audio) > 10 * 1024 * 1024:
        raise SpeechError("Chatterbox returned an oversized audio response.")
    if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
        raise SpeechError("Chatterbox did not return WAV audio.")
    return audio


def play_audio(audio: bytes, stop: Event, muted: Event) -> bool:
    if stop.is_set() or muted.is_set():
        return False
    try:
        import sounddevice as sd
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(audio)) as source:
            if source.frames / source.samplerate > 60 or source.channels > 2:
                raise SpeechError("Narration audio is too long or has unsupported channels.")
            samples = source.read(dtype="float32", always_2d=True)
            sample_rate = source.samplerate
        if stop.is_set() or muted.is_set():
            return False
        sd.play(samples, sample_rate)
        try:
            while sd.get_stream().active:
                if stop.wait(0.05) or muted.is_set():
                    return False
        finally:
            sd.stop()
        return True
    except SpeechError:
        raise
    except Exception:
        raise SpeechError("Audio playback failed. Check sounddevice/soundfile and the output device.") from None
