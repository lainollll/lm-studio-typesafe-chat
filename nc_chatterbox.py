"""Isolated access to NeuralCompanion's installed Chatterbox service."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from collections.abc import Iterator
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
from threading import Event, Thread
import time

from chatterbox_client import SpeechError


@dataclass(frozen=True)
class NCConfig:
    root: str = ""
    python: str = ""
    device: str = "auto"
    reference: str = ""
    local: bool = False


def find_nc_python(root: str) -> str:
    folder = Path(root)
    for relative in (".venv/Scripts/python.exe", "venv/Scripts/python.exe", "python/python.exe", "runtime/python/python.exe"):
        candidate = folder / relative
        if candidate.is_file():
            return str(candidate)
    return ""


class NCWorker:
    def __init__(self, config: NCConfig, stop: Event) -> None:
        self.config, self.stop = config, stop
        self.process: subprocess.Popen | None = None
        self.responses: Queue[dict] = Queue(maxsize=2)
        self._closed = Event()

    def start(self) -> None:
        root = Path(__file__).resolve().parent if self.config.local else Path(self.config.root)
        python = str(root / ".venv-chatterbox/Scripts/python.exe") if self.config.local else self.config.python.strip() or find_nc_python(str(root))
        if self.config.local and not Path(python).is_file():
            raise SpeechError("Project Chatterbox is not installed. Run install_chatterbox.ps1 first.")
        if not self.config.local and (not (root / "addons/chatterbox_tts/service.py").is_file() or not python or not Path(python).is_file()):
            raise SpeechError("Choose the installed NC folder and its Python executable in Speech settings.")
        if self.config.device not in ("auto", "cpu", "cuda"):
            raise SpeechError("Choose auto, cpu or cuda for NC Chatterbox.")
        environment = dict(os.environ)
        environment.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1", PYTHONIOENCODING="utf-8")
        self.process = subprocess.Popen(
            [python, "-B", "-u", str(Path(__file__).with_name("nc_chatterbox_worker.py")), str(root.resolve()), self.config.device]
            + (["--local"] if self.config.local else []),
            cwd=str(root.resolve()), env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        Thread(target=self._read, daemon=True).start()
        result = self._wait(180)
        if not result.get("ready"):
            raise SpeechError(result.get("error", "NC Chatterbox failed to initialize."))

    def _read(self) -> None:
        assert self.process and self.process.stdout
        output = self.process.stdout
        try:
            while True:
                line = output.readline(16 * 1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 16 * 1024 * 1024:
                    break
                value = json.loads(line)
                if not isinstance(value, dict):
                    break
                if not self._enqueue(value):
                    break
        except (OSError, ValueError):
            pass
        finally:
            output.close()
        self._enqueue({"error": "Chatterbox worker stopped unexpectedly."})

    def _enqueue(self, value: dict) -> bool:
        while not self.stop.is_set() and not self._closed.is_set():
            try:
                self.responses.put(value, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def stream(self, text: str) -> Iterator[bytes]:
        """Yield phrase WAVs immediately; consume to completion before another request."""
        if not self.process or not self.process.stdin:
            raise SpeechError("Chatterbox is not loaded.")
        try:
            self.process.stdin.write(json.dumps({"text": text, "reference": self.config.reference, "stream": True}) + "\n")
            self.process.stdin.flush()
            while True:
                result = self._wait(30)
                if result.get("error"):
                    raise SpeechError(result["error"])
                if result.get("done"):
                    return
                audio = base64.b64decode(result["audio"], validate=True)
                if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
                    raise ValueError
                yield audio
        except (OSError, KeyError, ValueError):
            raise SpeechError("Invalid audio from the Chatterbox worker.") from None

    def _wait(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while not self.stop.is_set() and time.monotonic() < deadline:
            try:
                return self.responses.get(timeout=0.1)
            except Empty:
                pass
        self.close()
        raise SpeechError("Chatterbox stopped or timed out.")

    def generate(self, text: str) -> bytes:
        if not self.process or not self.process.stdin:
            raise SpeechError("Chatterbox is not loaded.")
        try:
            self.process.stdin.write(json.dumps({"text": text, "reference": self.config.reference}) + "\n")
            self.process.stdin.flush()
            result = self._wait(30)
            if result.get("error"):
                raise SpeechError(result["error"])
            audio = base64.b64decode(result["audio"], validate=True)
            if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
                raise ValueError
            return audio
        except (OSError, KeyError, ValueError):
            raise SpeechError("Invalid audio from the Chatterbox worker.") from None

    def close(self) -> None:
        self._closed.set()
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            self.process = None
