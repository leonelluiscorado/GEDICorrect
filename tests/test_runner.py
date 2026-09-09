import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gedicorrect.config import CorrectionConfig
from gedicorrect.runner import run_correction


class FakeGEDICorrect:
    last_arguments = None
    last_simulation_arguments = None

    def __init__(self, **arguments):
        FakeGEDICorrect.last_arguments = arguments

    def simulate(self, **arguments):
        FakeGEDICorrect.last_simulation_arguments = arguments
        return ["/tmp/FOOTPRINT_granule.gpkg"]


class RunnerTests(unittest.TestCase):
    def test_runner_forwards_all_core_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            las_dir = root / "als"
            input_dir = root / "input"
            las_dir.mkdir()
            input_dir.mkdir()
            (las_dir / "tile.las").touch()
            (input_dir / "granule.gpkg").touch()

            fake_module = types.ModuleType("gedicorrect.correct")
            fake_module.GEDICorrect = FakeGEDICorrect
            config = CorrectionConfig(
                las_dir=str(las_dir),
                granules_dir=str(input_dir),
                out_dir=str(root / "output"),
                time_window=0.08,
                parallel=True,
                n_processes=3,
            )

            with patch.dict(sys.modules, {"gedicorrect.correct": fake_module}):
                result = run_correction(config, check_environment=False)

            self.assertEqual(FakeGEDICorrect.last_arguments["time_window"], 0.08)
            self.assertTrue(FakeGEDICorrect.last_arguments["use_parallel"])
            self.assertEqual(FakeGEDICorrect.last_arguments["n_processes"], 3)
            self.assertEqual(FakeGEDICorrect.last_simulation_arguments, {"grid_size": 15, "grid_step": 1})
            self.assertEqual(result.output_files, ("/tmp/FOOTPRINT_granule.gpkg",))


if __name__ == "__main__":
    unittest.main()
