import base64
import io
import unittest
from threading import Event
from types import SimpleNamespace

from nc_chatterbox import NCConfig, NCWorker
import nc_chatterbox_worker as child


class ChunkTests(unittest.TestCase):
    def test_long_description_is_split_without_losing_words(self):
        text = "A person enters the room. " + " ".join(["word"] * 35)
        split = getattr(child, "speech_chunks", lambda value: [value])
        chunks = split(text)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(" ".join(chunks), text)
        self.assertTrue(all(len(chunk.split()) <= 24 for chunk in chunks))
        self.assertEqual(chunks[0], "A person enters the room.")

    def test_punctuation_and_abbreviations_preserve_phrase_intonation(self):
        text = 'Dr. Smith asks, "Is that 3.14?" Yes! A person opens the door, then waves.'
        chunks = child.speech_chunks(text)
        self.assertEqual(chunks, ['Dr. Smith asks, "Is that 3.14?"', 'Yes!',
                                  'A person opens the door,', 'then waves.'])

    def test_sentence_over_twelve_words_stays_whole(self):
        text = "The person standing near the open window looks across the room at the red chair."
        self.assertEqual(child.speech_chunks(text), [text])

    def test_first_audio_available_before_end_of_response(self):
        worker = NCWorker(NCConfig(), Event())
        worker.process = SimpleNamespace(stdin=io.StringIO())
        wav = b"RIFF0000WAVEfirst"
        worker.responses.put({"audio": base64.b64encode(wav).decode()})
        worker.responses.put({"done": True})
        self.assertTrue(hasattr(worker, "stream"), "Need incremental audio API")
        chunks = worker.stream("A short sentence.")
        self.assertEqual(next(chunks), wav)
        self.assertEqual(worker.responses.qsize(), 1)
        self.assertEqual(list(chunks), [])

    def test_stream_error_is_reported(self):
        worker = NCWorker(NCConfig(), Event())
        worker.process = SimpleNamespace(stdin=io.StringIO())
        worker.responses.put({"error": "Synthesis failed"})
        self.assertTrue(hasattr(worker, "stream"), "Need incremental audio API")
        from chatterbox_client import SpeechError
        with self.assertRaises(SpeechError):
            list(worker.stream("Hello"))


if __name__ == "__main__":
    unittest.main()
