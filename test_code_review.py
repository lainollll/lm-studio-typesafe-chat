import unittest
from dataclasses import replace
from unittest.mock import patch

from chat_backend import ApiError, Settings, chat
from code_review import check_python


class SyntaxTests(unittest.TestCase):
    def test_valid_code_is_compiled_without_execution(self):
        result = check_python('```python\nraise RuntimeError("Must not execute")\n```')
        self.assertFalse(result.issues)
        self.assertIn("not executed", result.summary)

    def test_invalid_syntax_reports_location(self):
        result = check_python('```python\ndef broken(\n```')
        self.assertTrue(result.issues)
        self.assertIn("line 1", result.summary)

    def test_context_errors_are_caught(self):
        self.assertTrue(check_python('```python\nreturn 3\n```').issues)

    def test_missing_or_unclosed_python_fence_is_not_a_pass(self):
        for text in ("No code", "```python\nx = 1", "```javascript\nlet x = 1;\n```", "```python\n\n```"):
            with self.subTest(text=text):
                self.assertTrue(check_python(text).issues)

    def test_all_python_blocks_are_checked(self):
        result = check_python('```python\nx=1\n```\n```py\nif True print(2)\n```')
        self.assertTrue(result.issues)
        self.assertIn("block 2", result.summary)


class CodeFlowTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings("http://localhost:1234/v1", "model", "secret", code_mode=True)

    @patch("chat_backend.request_json")
    def test_jev_code_issue_causes_one_repair(self, request):
        request.side_effect = [
            {"choices": [{"message": {"content": "```python\nprint(1/0)\n```"}}]},
            {"answers": {name: {"type": "noul", "noul": value} for name, value in
                         [("requirements", 0.1), ("runtime", 0.99), ("edge_cases", 0.2)]}},
            {"choices": [{"message": {"content": "```python\nprint(1)\n```"}}]},
        ]
        result = chat(self.settings, [], "Print 1")
        self.assertIn("print(1)", result.reply)
        self.assertIn("not executed", result.analysis)
        self.assertEqual(request.call_count, 3)
        review = request.call_args_list[1].kwargs["payload"]
        self.assertIn("runtime", review["questions"])
        self.assertNotIn("mood_mismatch", review["questions"])

    @patch("chat_backend.request_json")
    def test_syntax_repair_works_when_jev_is_unavailable(self, request):
        request.side_effect = [
            {"choices": [{"message": {"content": "```python\nif\n```"}}]},
            ApiError("HTTP 529"),
            {"choices": [{"message": {"content": "```python\nprint(1)\n```"}}]},
        ]
        result = chat(self.settings, [], "Print 1")
        self.assertIn("print(1)", result.reply)
        self.assertIn("529", result.analysis)
        self.assertIn("Final Python syntax: PASS", result.analysis)

    @patch("chat_backend.request_json")
    def test_review_off_still_checks_syntax_without_repair(self, request):
        request.return_value = {"choices": [{"message": {"content": "```python\nif\n```"}}]}
        result = chat(replace(self.settings, refine=False), [], "Print 1")
        self.assertIn("Final Python syntax: FAIL", result.analysis)
        self.assertEqual(request.call_count, 1)

    @patch("chat_backend.request_json")
    def test_bad_repair_is_reported_without_another_attempt(self, request):
        request.side_effect = [
            {"choices": [{"message": {"content": "```python\nif\n```"}}]},
            {"choices": [{"message": {"content": "```python\nreturn 1\n```"}}]},
        ]
        result = chat(replace(self.settings, typesafe_key=""), [], "Print 1")
        self.assertIn("Final Python syntax: FAIL", result.analysis)
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
