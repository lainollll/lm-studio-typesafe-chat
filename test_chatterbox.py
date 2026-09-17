import io
import json
from threading import Event
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from chatterbox_client import ChatterboxConfig, SpeechError, request_audio, play_audio


class ChatterboxTests(unittest.TestCase):
    @patch("chatterbox_client.urlopen")
    def test_native_request_uses_text_and_wav_without_leaking_key(self, open_url):
        response = open_url.return_value.__enter__.return_value
        response.read.return_value = b"RIFF0000WAVEdata"
        result = request_audio(ChatterboxConfig(endpoint="http://localhost:8004/tts", key="secret"), "Hello")
        self.assertEqual(result, b"RIFF0000WAVEdata")
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["text"], "Hello")
        self.assertEqual(payload["output_format"], "wav")
        self.assertNotIn("secret", str(payload))

    @patch("chatterbox_client.urlopen")
    def test_openai_endpoint_uses_input_and_voice(self, open_url):
        open_url.return_value.__enter__.return_value.read.return_value = b"RIFF0000WAVEdata"
        request_audio(ChatterboxConfig(endpoint="http://localhost:8004/v1/audio/speech", voice="voice.wav"), "Hello")
        payload = json.loads(open_url.call_args.args[0].data)
        self.assertEqual(payload["input"], "Hello")
        self.assertEqual(payload["voice"], "voice.wav")
        self.assertEqual(payload["response_format"], "wav")

    @patch("chatterbox_client.urlopen")
    def test_non_wav_response_is_rejected(self, open_url):
        open_url.return_value.__enter__.return_value.read.return_value = b'{"error":"failed"}'
        with self.assertRaises(SpeechError):
            request_audio(ChatterboxConfig(), "Hello")

    @patch("chatterbox_client.urlopen")
    def test_http_error_does_not_reveal_server_body(self, open_url):
        open_url.side_effect = HTTPError("http://localhost", 401, "secret", {}, io.BytesIO(b"secret"))
        with self.assertRaises(SpeechError) as caught:
            request_audio(ChatterboxConfig(key="secret"), "Hello")
        self.assertNotIn("secret", str(caught.exception))

    def test_stop_prevents_playback(self):
        stop, mute = Event(), Event()
        stop.set()
        self.assertFalse(play_audio(b"invalid", stop, mute))


if __name__ == "__main__":
    unittest.main()
