import io
import unittest
import wave
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from chatterbox_client import SpeechError, play_audio


def silent_wav(seconds: int = 1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 24000 * seconds)
    return output.getvalue()


class PlaybackTests(unittest.TestCase):
    def test_real_wav_decodes_and_reaches_playback(self):
        with patch("sounddevice.play") as play, patch("sounddevice.get_stream", return_value=SimpleNamespace(active=False)), patch("sounddevice.stop"):
            self.assertTrue(play_audio(silent_wav(), Event(), Event()))
            samples, rate = play.call_args.args
            self.assertEqual(samples.shape, (24000, 1))
            self.assertEqual(rate, 24000)

    def test_oversized_duration_is_rejected_before_playback(self):
        with patch("sounddevice.play") as play:
            with self.assertRaisesRegex(SpeechError, "too long"):
                play_audio(silent_wav(61), Event(), Event())
            play.assert_not_called()


if __name__ == "__main__":
    unittest.main()
