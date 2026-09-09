import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from gedicorrect.cli import build_parser, main
from gedicorrect.system import EnvironmentReport


class CommandLineTests(unittest.TestCase):
    def test_run_parser_accepts_legacy_underscore_flags(self):
        arguments = build_parser().parse_args([
            "run",
            "--granules_dir", "input",
            "--las_dir", "als",
            "--out_dir", "output",
            "--time_window", "0.05",
        ])
        self.assertEqual(arguments.granules_dir, "input")
        self.assertEqual(arguments.time_window, 0.05)

    def test_prepare_parser_accepts_alignment_paths(self):
        arguments = build_parser().parse_args([
            "prepare",
            "align",
            "--l1b-dir", "l1b",
            "--l2a-dir", "l2a",
            "--out-dir", "merged",
        ])
        self.assertEqual(arguments.prepare_command, "align")
        self.assertEqual(arguments.out_dir, "merged")

    @patch("gedicorrect.cli.inspect_environment")
    def test_check_reports_missing_executables(self, inspect_environment):
        inspect_environment.return_value = EnvironmentReport(
            cpu_count=4,
            executables={"gediRat": None, "gediMetric": None},
        )
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["check"])

        self.assertEqual(return_code, 1)
        self.assertIn("setup required", output.getvalue())


if __name__ == "__main__":
    unittest.main()
