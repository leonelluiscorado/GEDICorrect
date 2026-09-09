import tempfile
import unittest
from pathlib import Path

from gedicorrect.config import CorrectionConfig


class CorrectionConfigTests(unittest.TestCase):
    def make_paths(self):
        temporary_directory = tempfile.TemporaryDirectory()
        root = Path(temporary_directory.name)
        las_dir = root / "als"
        granules_dir = root / "input"
        las_dir.mkdir()
        granules_dir.mkdir()
        (las_dir / "tile.las").touch()
        (granules_dir / "granule.gpkg").touch()
        return temporary_directory, las_dir, granules_dir, root / "output"

    def test_valid_directory_configuration(self):
        temporary_directory, las_dir, granules_dir, output_dir = self.make_paths()
        self.addCleanup(temporary_directory.cleanup)

        config = CorrectionConfig(
            las_dir=str(las_dir),
            granules_dir=str(granules_dir),
            out_dir=str(output_dir),
        )

        self.assertIs(config, config.validate())

    def test_requires_exactly_one_gedi_input(self):
        config = CorrectionConfig(las_dir="als", out_dir="output")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            config.validate(check_paths=False)

    def test_rejects_incompatible_correlations(self):
        config = CorrectionConfig(
            las_dir="als",
            granules_dir="input",
            out_dir="output",
            criteria="wave_pearson wave_spearman",
        )
        with self.assertRaisesRegex(ValueError, "either wave_pearson"):
            config.validate(check_paths=False)

    def test_all_criteria_uses_a_compatible_correlation_set(self):
        config = CorrectionConfig(
            las_dir="als",
            granules_dir="input",
            out_dir="output",
            criteria="all",
        )
        self.assertIs(config, config.validate(check_paths=False))

    def test_random_mode_requires_footprint_correction(self):
        config = CorrectionConfig(
            las_dir="als",
            granules_dir="input",
            out_dir="output",
            mode="orbit",
            random=True,
        )
        with self.assertRaisesRegex(ValueError, "footprint mode"):
            config.validate(check_paths=False)

    def test_cli_serialization_preserves_time_window(self):
        config = CorrectionConfig(
            las_dir="/data/als",
            granules_dir="/data/input",
            out_dir="/data/output",
            time_window=0.08,
            parallel=True,
        )
        arguments = config.to_cli_args()
        index = arguments.index("--time-window")
        self.assertEqual(arguments[index + 1], "0.08")
        self.assertIn("--parallel", arguments)


if __name__ == "__main__":
    unittest.main()
