import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gedicorrect import jobs


class FakeProcess:
    def __init__(self, pid=4242):
        self.pid = pid
        self.return_code = None

    def poll(self):
        return self.return_code


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.state_path = self.root / "state" / "job.json"
        self.environment = patch.dict(os.environ, {jobs.JOB_STATE_ENV: str(self.state_path)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        jobs._PROCESS_HANDLES.clear()
        self.addCleanup(jobs._PROCESS_HANDLES.clear)

    def start_fake_job(self):
        process = FakeProcess()
        command = [sys.executable, "-m", "gedicorrect", "run"]
        with patch.object(jobs.subprocess, "Popen", return_value=process):
            record = jobs.start_job(command, self.root / "output")
        return process, record

    def test_running_job_is_persisted_and_restored(self):
        _, record = self.start_fake_job()
        self.assertTrue(self.state_path.is_file())
        self.assertTrue(record.running)

        jobs._PROCESS_HANDLES.clear()
        with patch.object(jobs, "_same_process_is_running", return_value=True):
            restored = jobs.get_job()

        self.assertEqual(restored.pid, record.pid)
        self.assertTrue(restored.running)

    def test_completed_process_updates_persisted_status(self):
        process, _ = self.start_fake_job()
        process.return_code = 0

        completed = jobs.get_job()

        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.return_code, 0)
        self.assertEqual(jobs.get_job().status, "completed")

    def test_cancelled_process_updates_persisted_status(self):
        _, record = self.start_fake_job()

        with patch.object(jobs.os, "killpg") as kill_process_group:
            cancelled = jobs.cancel_job(record)

        kill_process_group.assert_called_once_with(record.pid, jobs.signal.SIGTERM)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(jobs.get_job().status, "cancelled")

    def test_terminal_progress_updates_collapse_to_latest_line(self):
        output = (
            "Setup complete\n"
            "\rProcessing footprints:   0%|          | 0/10"
            "\rProcessing footprints:  50%|#####     | 5/10"
            "\rProcessing footprints: 100%|##########| 10/10\n"
            "Correction complete\n"
        )

        normalized = jobs._normalize_terminal_output(output)

        self.assertEqual(
            normalized,
            "Setup complete\nProcessing footprints: 100%|##########| 10/10\nCorrection complete\n",
        )

    def test_terminal_formatting_codes_are_removed(self):
        output = "\x1b[32mReady\x1b[0m\r\n\rRunning 10%\rRunning 20%"

        self.assertEqual(jobs._normalize_terminal_output(output), "Ready\nRunning 20%")


if __name__ == "__main__":
    unittest.main()
