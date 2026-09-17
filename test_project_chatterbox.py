import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch, MagicMock

from nc_chatterbox import NCConfig, NCWorker


class ProjectChatterboxTests(unittest.TestCase):
    def test_project_worker_uses_project_python_without_nc(self):
        self.assertIn("local", NCConfig.__dataclass_fields__)
        worker = NCWorker(NCConfig(root="invalid NC folder", python="invalid NC python", local=True), Event())
        with patch("nc_chatterbox.subprocess.Popen") as launch, patch("nc_chatterbox.Thread"), patch.object(worker, "_wait", return_value={"ready": True}):
            worker.start()
        command = launch.call_args.args[0]
        self.assertEqual(Path(command[0]).parent.parent.name, ".venv-chatterbox")
        self.assertEqual(command[-1], "--local")
        self.assertNotIn("invalid NC folder", command)


if __name__ == "__main__":
    unittest.main()
