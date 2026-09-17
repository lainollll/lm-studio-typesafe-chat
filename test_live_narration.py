import time
import tempfile
from pathlib import Path
from threading import Event, Thread
import unittest
from unittest.mock import patch

from chat_backend import ApiError, Settings
from live_narration import Frame, LatestFrame, LiveNarrator, LiveOptions, describe_frame, should_announce, is_fresh


class LiveTests(unittest.TestCase):
    @patch("live_narration.request_json")
    def test_slider_change_during_review_uses_latest_threshold(self, request):
        narrator = LiveNarrator(Settings("http://localhost", "model", "key"), LiveOptions(sensitivity=0))
        def reply(*args, **kwargs):
            narrator.set_sensitivity(100)
            return {"answers": {"new_information": {"type": "noul", "noul": 0.5}}}
        request.side_effect = reply
        announce, note = should_announce(narrator.settings, "Sitting.", "Standing.", narrator.get_sensitivity)
        self.assertTrue(announce)
        self.assertIn("25%", note)

    @patch("live_narration.request_json")
    def test_sensitivity_changes_announcement_decision(self, request):
        request.return_value = {"answers": {"new_information": {"type": "noul", "noul": 0.5}}}
        for sensitivity, expected in [(0, False), (50, False), (100, True)]:
            with self.subTest(sensitivity=sensitivity):
                announce, note = should_announce(Settings("http://localhost", "model", "key"),
                                                 "A person sits.", "A person stands.", sensitivity=sensitivity)
                self.assertEqual(announce, expected)
                self.assertIn("threshold", note)

    def test_sensitivity_can_change_during_session(self):
        narrator = LiveNarrator(Settings("http://localhost", "model"), LiveOptions())
        self.assertEqual(narrator.get_sensitivity(), 50)
        narrator.set_sensitivity(90)
        self.assertEqual(narrator.get_sensitivity(), 90)
        narrator.set_sensitivity(200)
        self.assertEqual(narrator.get_sensitivity(), 100)

    @patch("live_narration.request_json")
    def test_truncated_model_output_is_not_spoken_as_complete(self, request):
        request.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "A person is"}}]}
        with self.assertRaises(ApiError):
            describe_frame(Settings("http://localhost", "chat", vision_model="vision"), Frame(1,time.monotonic(),b"frame"))

    def test_stop_discards_inflight_description(self):
        entered, release = Event(), Event()
        narrator = LiveNarrator(Settings("http://localhost", "chat", vision_model="vision"), LiveOptions())
        narrator.frames.put(Frame(1, time.monotonic(), b"frame"))

        def delayed(*args):
            entered.set()
            release.wait(2)
            return "Late description"

        with patch("live_narration.describe_frame", side_effect=delayed), patch("live_narration.should_announce") as judge:
            thread = Thread(target=narrator._infer)
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                narrator.stop()
            finally:
                release.set()
                thread.join(2)
            self.assertFalse(thread.is_alive())
            judge.assert_not_called()
            self.assertTrue(narrator.speech.empty())

    def test_video_file_source_decodes_frames_and_releases_file(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for value in (0, 60, 120, 180):
                    writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            narrator = LiveNarrator(Settings("http://localhost", "chat"), LiveOptions(source="file", file_path=str(path)))
            narrator._capture()
            self.assertTrue(narrator.ended.is_set())
            self.assertFalse(narrator.stop_event.is_set())
            self.assertTrue(narrator.frames.get().jpeg.startswith(b"\xff\xd8"))
            path.unlink()  # The capture handle must have been released, including on Windows.

    def test_latest_frame_replaces_backlog(self):
        frames = LatestFrame()
        frames.put(Frame(1, time.monotonic(), b"old"))
        frames.put(Frame(2, time.monotonic(), b"new"))
        self.assertEqual(frames.get().jpeg, b"new")

    def test_old_frame_is_not_current_narration(self):
        self.assertFalse(is_fresh(Frame(1, time.monotonic() - 30, b"old")))
        self.assertTrue(is_fresh(Frame(2, time.monotonic(), b"new")))

    @patch("live_narration.request_json")
    def test_exact_repeat_is_skipped_without_api_call(self, request):
        announce, _ = should_announce(Settings("http://localhost", "model", "key"), "A red car.", " A red car. ")
        self.assertFalse(announce)
        request.assert_not_called()

    @patch("live_narration.request_json")
    def test_jev_can_suppress_repeat_or_report_change(self, request):
        for score, expected in [(0.1, False), (0.9, True)]:
            request.return_value = {"answers": {"new_information": {"type": "noul", "noul": score}}}
            announce, _ = should_announce(Settings("http://localhost", "model", "key"), "A car is parked.", "A car drives away.")
            self.assertEqual(announce, expected)
            self.assertNotIn("image_url", str(request.call_args.kwargs["payload"]))

    @patch("live_narration.request_json", side_effect=ApiError("HTTP 529"))
    def test_jev_outage_keeps_new_description_available(self, request):
        announce, note = should_announce(Settings("http://localhost", "model", "key"), "", "A room.")
        self.assertTrue(announce)
        self.assertIn("529", note)


if __name__ == "__main__":
    unittest.main()
