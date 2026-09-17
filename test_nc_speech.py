import time
import unittest
from threading import Event
from unittest.mock import patch

from chat_backend import Settings
from chatterbox_client import SpeechError
from live_narration import LiveNarrator, LiveOptions
from nc_chatterbox import NCConfig, NCWorker


class NCSpeechTests(unittest.TestCase):
    def narrator(self):
        return LiveNarrator(Settings("http://localhost:1234/v1", "model"), LiveOptions(speech_provider="NC Chatterbox"))

    def test_load_failure_falls_back_and_unblocks_capture(self):
        narrator = self.narrator()
        with patch("live_narration.NCWorker") as worker, patch.object(narrator, "_speak_windows") as fallback:
            worker.return_value.start.side_effect = SpeechError("Missing cached weights")
            narrator._speak()
            fallback.assert_called_once()
            worker.return_value.close.assert_called_once()
        self.assertTrue(narrator.speech_ready.is_set())
        self.assertIn("Switching to Windows", narrator.events.get()[1] if narrator.events.qsize() == 1 else list(narrator.events.queue)[-1][1])

    def test_stop_during_load_does_not_start_fallback_or_capture(self):
        narrator = self.narrator()
        def cancel():
            narrator.stop()
            raise SpeechError("Cancelled")
        with patch("live_narration.NCWorker") as worker, patch.object(narrator, "_speak_windows") as fallback, patch.object(narrator, "_capture_screen") as capture:
            worker.return_value.start.side_effect = cancel
            narrator._speak()
            narrator._capture()
            fallback.assert_not_called()
            capture.assert_not_called()

    def test_synthesis_that_becomes_stale_is_not_played(self):
        narrator = self.narrator()
        narrator.speech.put(("A circle", time.monotonic() - 19))
        def generate(text):
            narrator.stop()
            yield b"audio"
        with patch("live_narration.NCWorker") as worker, patch("live_narration.play_audio") as play:
            worker.return_value.stream.side_effect = generate
            with patch("live_narration.time.monotonic", side_effect=[time.monotonic(), time.monotonic() + 5]):
                narrator._speak()
            play.assert_not_called()

    def test_cancel_wait_closes_worker(self):
        stop = Event()
        stop.set()
        worker = NCWorker(NCConfig(), stop)
        with patch.object(worker, "close") as close:
            with self.assertRaises(SpeechError):
                worker._wait(180)
            close.assert_called_once()

    def test_first_chunk_plays_before_next_chunk_is_requested(self):
        narrator = self.narrator()
        narrator.speech.put(("First sentence. Next sentence.", time.monotonic()))
        events = []
        def stream(text):
            events.append("generate first")
            yield b"first"
            events.append("generate second")
            yield b"second"
            narrator.stop()
        def play(audio, stop, muted):
            events.append(audio.decode())
            return True
        with patch("live_narration.NCWorker") as worker, patch("live_narration.play_audio", side_effect=play):
            worker.return_value.stream.side_effect = stream
            narrator._speak()
        self.assertEqual(events, ["generate first", "first", "generate second", "second"])

    def test_mute_abandons_remaining_chunks_even_after_unmute(self):
        narrator = self.narrator()
        narrator.speech.put(("First. Second. Third.", time.monotonic()))
        heard = []
        def stream(text):
            yield b"first"
            narrator.set_muted(True)
            yield b"second"
            narrator.set_muted(False)
            yield b"third"
            narrator.stop()
        def play(audio, stop, muted):
            heard.append(audio)
            return True
        with patch("live_narration.NCWorker") as worker, patch("live_narration.play_audio", side_effect=play):
            worker.return_value.stream.side_effect = stream
            narrator._speak()
        self.assertEqual(heard, [b"first"])


if __name__ == "__main__":
    unittest.main()
