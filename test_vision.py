import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_backend import ApiError, Settings, chat
from vision_image import image_data_url


class VisionTests(unittest.TestCase):
    def test_rejects_non_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fake.png"
            path.write_text("not an image")
            with self.assertRaises(ValueError):
                image_data_url(str(path))

    @patch("chat_backend.image_data_url", return_value="data:image/png;base64,IMAGE_BYTES")
    @patch("chat_backend.request_json")
    def test_image_goes_only_to_vision_and_description_reaches_jev(self, request, image):
        request.side_effect = [
            {"choices": [{"message": {"content": "A red cup on a table."}}]},
            {"answers": {"tone": {"type": "choice", "choice": "neutral", "confidence": 1}}},
            {"choices": [{"message": {"content": "The cup appears red."}}]},
            {"answers": {name: {"type": "noul", "noul": 0.1} for name in
                         ("stiff", "repetitive", "off_topic", "mood_mismatch", "visual_grounding")}},
        ]
        settings = Settings("http://localhost:1234/v1", "chat-model", "secret",
                            vision_model="lfm-vl", vision_base_url="http://localhost:4321/v1")
        result = chat(settings, [], "What color is the cup?", image_path="test.png")
        calls = request.call_args_list
        self.assertEqual(calls[0].args[0], "http://localhost:4321/v1/chat/completions")
        self.assertEqual(calls[0].kwargs["payload"]["model"], "lfm-vl")
        self.assertIn("IMAGE_BYTES", json.dumps(calls[0].kwargs["payload"]))
        for call in calls[1:]:
            self.assertNotIn("IMAGE_BYTES", json.dumps(call.kwargs["payload"]))
        self.assertIn("A red cup", json.dumps(calls[2].kwargs["payload"]))
        self.assertEqual(calls[3].kwargs["payload"]["state"]["vision_description"], "A red cup on a table.")
        self.assertEqual(result.vision_description, "A red cup on a table.")
        self.assertIn("A red cup", result.user_context)

    @patch("chat_backend.image_data_url", return_value="data:image/png;base64,IMAGE")
    @patch("chat_backend.request_json", side_effect=ApiError("Vision unavailable"))
    def test_failed_vision_does_not_generate_an_ungrounded_reply(self, request, image):
        with self.assertRaises(ApiError):
            chat(Settings("http://localhost:1234/v1", "chat", vision_model="vision"), [], "Describe", image_path="a.png")
        self.assertEqual(request.call_count, 1)

    @patch("chat_backend.request_json")
    def test_attachment_requires_selected_vision_model(self, request):
        with self.assertRaises(ApiError):
            chat(Settings("http://localhost:1234/v1", "chat"), [], "Describe", image_path="a.png")
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
